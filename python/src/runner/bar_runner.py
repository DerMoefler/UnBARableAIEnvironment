import time
import wandb
import numpy as np
from functools import reduce
import torch
from onpolicy.runner.shared.base_runner import Runner

# ─────────────────────────────────────────────────────────────────────────────
# ERKLÄRUNG: _t2n (Tensor to Numpy)
# Diese Hilfsfunktion steht außerhalb der Klasse, weil sie keinen Zustand
# braucht. Sie macht drei Dinge:
#   1. .detach()  → trennt den Tensor vom PyTorch-Gradienten-Graph
#   2. .cpu()     → verschiebt ihn vom GPU-Speicher in den RAM
#   3. .numpy()   → konvertiert ihn in ein NumPy-Array für den Buffer
# ─────────────────────────────────────────────────────────────────────────────
def _t2n(x):
    return x.detach().cpu().numpy()


class BARRunner(Runner):
    """
    Runner für Beyond All Reason (BAR).

    Jeder Agent steuert eine einzelne Einheit.
    Observation: eigene Einheiten (Position, HP, Typ) + feindliche Einheiten.
    Aktionen: diskret (Move, Attack, Stop, ...).

    Erbt von Runner (base_runner.py), der folgendes bereitstellt:
        self.envs              → die parallelen BAR-Environments
        self.eval_envs         → separate Envs nur für Evaluation
        self.buffer            → SharedReplayBuffer (speichert Rollout-Daten)
        self.trainer           → R_MAPPO (führt PPO-Updates durch)
        self.policy            → Actor + Critic Netze
        self.num_agents        → Anzahl Agenten (= Anzahl Einheiten)
        self.episode_length    → maximale Steps pro Episode
        self.n_rollout_threads → Anzahl paralleler Spielinstanzen
    """

    def __init__(self, config):
        # ─────────────────────────────────────────────────────────────────────
        # ERKLÄRUNG: super().__init__(config)
        # Ruft den Konstruktor der Elternklasse (Runner) auf.
        # Dort werden alle self.* Attribute gesetzt (envs, buffer, trainer ...)
        # und die Policy + der Algorithmus initialisiert.
        # ─────────────────────────────────────────────────────────────────────
        super(BARRunner, self).__init__(config)

    # ─────────────────────────────────────────────────────────────────────────
    # ERKLÄRUNG: run()
    # Die Haupttrainingsschleife. Wird einmal aufgerufen und läuft bis das
    # gesamte Trainingsbudget (num_env_steps) verbraucht ist.
    # Ablauf pro Episode:
    #   1. collect() → Rollout sammeln (Daten in Buffer schreiben)
    #   2. compute() → Returns/Advantages berechnen (GAE)
    #   3. train()   → PPO-Update auf dem Buffer
    #   4. Logging + Evaluation
    # ─────────────────────────────────────────────────────────────────────────
    def run(self):
        self.warmup()

        start = time.time()

        # Gesamtanzahl Episoden = Budget / Steps pro Episode / parallele Envs
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        # ─────────────────────────────────────────────────────────────────────
        # ERKLÄRUNG: BAR-spezifisches Tracking
        # Wir tracken gewonnene und gespielte Spiele inkrementell,
        # um die Win-Rate nur über das letzte Log-Intervall zu berechnen
        # (nicht kumulativ über das gesamte Training).
        # last_battles_* speichert den Stand vom letzten Log-Zeitpunkt.
        # ─────────────────────────────────────────────────────────────────────
        last_battles_game = np.zeros(self.n_rollout_threads, dtype=np.float32)
        last_battles_won = np.zeros(self.n_rollout_threads, dtype=np.float32)

        for episode in range(episodes):

            # ─────────────────────────────────────────────────────────────────
            # ERKLÄRUNG: Linear Learning Rate Decay
            # Der Lernrate wird über die Zeit kleiner — am Anfang lernt das
            # Netz groß und exploriert, gegen Ende werden nur noch kleine
            # Korrekturen gemacht. Verhindert Overshooting am Ende des Trainings.
            # ─────────────────────────────────────────────────────────────────
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            # ─────────────────────────────────────────────────────────────────
            # ERKLÄRUNG: Rollout-Schleife
            # Für jeden Step der Episode:
            #   collect(step) → Policy sampelt Aktionen aus dem Buffer
            #   envs.step()   → Aktionen werden im Spiel ausgeführt
            #   insert()      → neue Obs, Rewards etc. in Buffer schreiben
            # ─────────────────────────────────────────────────────────────────
            for step in range(self.episode_length):
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)

                # Aktionen ans Spiel schicken, neue Observation empfangen
                # obs:               [n_threads, n_agents, obs_dim]
                # share_obs:         [n_threads, n_agents, state_dim]
                # rewards:           [n_threads, n_agents, 1]
                # dones:             [n_threads, n_agents]  (True = Einheit tot/Episode Ende)
                # infos:             Liste mit BAR-spezifischen Infos pro Thread
                # available_actions: [n_threads, n_agents, act_dim] (legale Aktionen)
                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions)

                data = (obs, share_obs, rewards, dones, infos, available_actions,
                        values, actions, action_log_probs,
                        rnn_states, rnn_states_critic)

                self.insert(data)

            # ─────────────────────────────────────────────────────────────────
            # ERKLÄRUNG: compute() und train()
            # compute() → berechnet den letzten Value-Schätzwert und rechnet
            #             daraus rückwärts durch den Rollout die GAE-Returns
            # train()   → führt mehrere PPO-Epochen auf dem Buffer durch:
            #             Forward-Pass → Loss → Backprop → Gewichte updaten
            # ─────────────────────────────────────────────────────────────────
            self.compute()
            train_infos = self.train()

            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            # Modell speichern
            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save()

            # ─────────────────────────────────────────────────────────────────
            # ERKLÄRUNG: Logging
            # Wir loggen BAR-spezifische Metriken:
            #   - battles_won / battles_game → kommen aus den infos der C++ API
            #   - incre_win_rate             → Win-Rate seit letztem Log-Intervall
            #   - dead_ratio                 → Anteil toter Einheiten im Rollout
            #
            # ANPASSUNG FÜR BAR:
            # Das info-Dict muss von deinem BAR-Env-Wrapper befüllt werden.
            # Du brauchst mindestens:
            #   info[0]['battles_won']  → int, kumulativ gewonnene Spiele
            #   info[0]['battles_game'] → int, kumulativ gespielte Spiele
            # ─────────────────────────────────────────────────────────────────
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n BAR Algo {} Exp {} updates {}/{} episodes, "
                      "total num timesteps {}/{}, FPS {}.\n"
                      .format(self.algorithm_name,
                              self.experiment_name,
                              episode,
                              episodes,
                              total_num_steps,
                              self.num_env_steps,
                              int(total_num_steps / (end - start))))

                # BAR Win-Rate Tracking
                battles_won = []
                battles_game = []
                incre_battles_won = []
                incre_battles_game = []

                for i, info in enumerate(infos):
                    if 'battles_won' in info[0].keys():
                        battles_won.append(info[0]['battles_won'])
                        incre_battles_won.append(info[0]['battles_won'] - last_battles_won[i])
                    if 'battles_game' in info[0].keys():
                        battles_game.append(info[0]['battles_game'])
                        incre_battles_game.append(info[0]['battles_game'] - last_battles_game[i])

                incre_win_rate = (np.sum(incre_battles_won) / np.sum(incre_battles_game)
                                  if np.sum(incre_battles_game) > 0 else 0.0)
                print("incre win rate is {}.".format(incre_win_rate))

                if self.use_wandb:
                    wandb.log({"incre_win_rate": incre_win_rate}, step=total_num_steps)
                else:
                    self.writter.add_scalars("incre_win_rate",
                                             {"incre_win_rate": incre_win_rate},
                                             total_num_steps)

                last_battles_game = battles_game
                last_battles_won = battles_won

                # ─────────────────────────────────────────────────────────────
                # ERKLÄRUNG: dead_ratio
                # active_masks ist 0 für tote Agenten, 1 für lebende.
                # dead_ratio = 1 - (Anteil lebender Agenten im gesamten Rollout)
                # Gibt an wie oft und wie viele Einheiten im Schnitt gestorben sind.
                # ─────────────────────────────────────────────────────────────
                train_infos['dead_ratio'] = (
                    1 - self.buffer.active_masks.sum()
                    / reduce(lambda x, y: x * y, list(self.buffer.active_masks.shape))
                )

                self.log_train(train_infos, total_num_steps)

            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    # ─────────────────────────────────────────────────────────────────────────
    # ERKLÄRUNG: warmup()
    # Wird einmal zu Beginn aufgerufen. Setzt alle Environments zurück und
    # schreibt die ersten Observations in Slot 0 des Buffers.
    # Slot 0 ist der Startpunkt — collect(0) liest ihn beim ersten Step.
    # ─────────────────────────────────────────────────────────────────────────
    def warmup(self):
        obs, share_obs, available_actions = self.envs.reset()
        # obs.shape         = [n_rollout_threads, n_agents, obs_dim]
        # share_obs.shape   = [n_rollout_threads, n_agents, state_dim]
        # available_actions = [n_rollout_threads, n_agents, n_actions]

        # Falls kein zentralisierter Critic gewünscht, sieht der Critic
        # dasselbe wie der Actor (kein globaler State)
        if not self.use_centralized_V:
            share_obs = obs

        # .copy() wichtig: verhindert dass der Buffer auf Env-Speicher zeigt
        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()
        self.buffer.available_actions[0] = available_actions.copy()

    # ─────────────────────────────────────────────────────────────────────────
    # ERKLÄRUNG: collect(step)
    # Liest die aktuelle Observation aus dem Buffer, schickt sie durch die
    # Policy und gibt Aktionen + Hilfswerte zurück.
    #
    # @torch.no_grad() → PyTorch baut keinen Gradienten-Graph auf.
    # Das spart Speicher, da wir hier nur sampeln, nicht trainieren.
    # ─────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()

        # ─────────────────────────────────────────────────────────────────────
        # ERKLÄRUNG: np.concatenate
        # buffer.obs[step] hat Shape [n_threads, n_agents, obs_dim]
        # z.B. [8, 5, 64] bei 8 parallelen Spielen und 5 Einheiten pro Team.
        # np.concatenate faltet Threads zusammen → [40, 64]
        # Die Policy bekommt einen einzigen großen Batch statt verschachtelter
        # Threads — das ist effizienter für das Netz.
        # ─────────────────────────────────────────────────────────────────────
        value, action, action_log_prob, rnn_state, rnn_state_critic \
            = self.trainer.policy.get_actions(
                np.concatenate(self.buffer.share_obs[step]),      # → Critic
                np.concatenate(self.buffer.obs[step]),            # → Actor
                np.concatenate(self.buffer.rnn_states[step]),     # GRU-Zustand Actor
                np.concatenate(self.buffer.rnn_states_critic[step]),  # GRU-Zustand Critic
                np.concatenate(self.buffer.masks[step]),          # 0 = Episode fertig
                np.concatenate(self.buffer.available_actions[step])   # legale Aktionen BAR
            )

        # ─────────────────────────────────────────────────────────────────────
        # ERKLÄRUNG: _t2n + np.split
        # Die Policy gibt PyTorch-Tensoren zurück (auf GPU).
        # _t2n: Tensor → NumPy (CPU)
        # np.split: [40, dim] zurück zu [8, 5, dim] (Threads wiederherstellen)
        # ─────────────────────────────────────────────────────────────────────
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_state), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_state_critic), self.n_rollout_threads))

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    # ─────────────────────────────────────────────────────────────────────────
    # ERKLÄRUNG: insert(data)
    # Verarbeitet den Output von envs.step() und schreibt alles in den Buffer.
    # Hier passiert die BAR-spezifische Masken-Logik:
    #
    # dones_env  → True wenn die gesamte Episode fertig ist (alle Einheiten tot
    #              oder Zeitlimit erreicht). Shape: [n_threads]
    # dones      → True wenn eine einzelne Einheit gestorben ist.
    #              Shape: [n_threads, n_agents]
    #
    # masks:        0 wenn Episode fertig → GRU-Zustand wird genullt
    # active_masks: 0 wenn einzelne Einheit tot → wird beim PPO-Loss ignoriert
    # bad_masks:    0 wenn Episode durch Timeout endete (kein echter done)
    #               → verhindert fehlerhafte Return-Berechnung am Episode-Ende
    # ─────────────────────────────────────────────────────────────────────────
    def insert(self, data):
        (obs, share_obs, rewards, dones, infos, available_actions,
         values, actions, action_log_probs, rnn_states, rnn_states_critic) = data

        # Episode komplett fertig? (alle Agenten done ODER Zeitlimit)
        dones_env = np.all(dones, axis=1)  # [n_threads]

        # GRU-Zustand nullen wenn Episode fertig → kein "Gedächtnis" aus
        # alter Episode soll in neue Episode übertragen werden
        rnn_states[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size),
            dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, *self.buffer.rnn_states_critic.shape[3:]),
            dtype=np.float32)

        # masks: 0 = Episode fertig, 1 = Episode läuft noch
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        # ─────────────────────────────────────────────────────────────────────
        # ERKLÄRUNG: active_masks — BAR-spezifisch
        # In BAR können einzelne Einheiten sterben während die Episode weiterläuft.
        # active_masks = 0 für tote Einheiten → der PPO-Loss ignoriert sie,
        # damit tote Einheiten das Training nicht verzerren.
        # Reihenfolge wichtig:
        #   1. Zuerst tote Einheiten auf 0 setzen
        #   2. Dann bei Episode-Ende alles auf 1 zurücksetzen (Reset)
        # ─────────────────────────────────────────────────────────────────────
        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(
            ((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        # ─────────────────────────────────────────────────────────────────────
        # ERKLÄRUNG: bad_masks
        # Wenn eine Episode durch Zeitlimit endet (nicht durch echten done),
        # ist der letzte Zustand kein echter Terminal-State.
        # bad_masks = 0 signalisiert dem Buffer: hier keinen Bootstrap-Wert
        # abschneiden → Return wird korrekt berechnet.
        #
        # ANPASSUNG FÜR BAR:
        # Dein BAR-Env-Wrapper muss in info[agent_id] ein Feld liefern:
        #   'bad_transition': True  → Episode endete durch Timeout
        #   'bad_transition': False → Episode endete durch echten done
        # ─────────────────────────────────────────────────────────────────────
        bad_masks = np.array([
            [[0.0] if info[agent_id]['bad_transition'] else [1.0]
             for agent_id in range(self.num_agents)]
            for info in infos
        ])

        if not self.use_centralized_V:
            share_obs = obs

        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic,
                           actions, action_log_probs, values, rewards,
                           masks, bad_masks, active_masks, available_actions)

    # ─────────────────────────────────────────────────────────────────────────
    # ERKLÄRUNG: log_train()
    # Loggt Trainingsmetriken nach jedem PPO-Update.
    # average_step_rewards = mittlerer Reward pro Step über alle Agenten
    # und Threads im letzten Rollout.
    # ─────────────────────────────────────────────────────────────────────────
    def log_train(self, train_infos, total_num_steps):
        train_infos["average_step_rewards"] = np.mean(self.buffer.rewards)
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)

    # ─────────────────────────────────────────────────────────────────────────
    # ERKLÄRUNG: eval()
    # Läuft deterministisch (kein Sampling, deterministic=True) um die echte
    # Leistung der Policy zu messen — ohne Explorationsrauschen.
    # Tracked: Win-Rate und durchschnittliche Episode-Rewards.
    #
    # ANPASSUNG FÜR BAR:
    # eval_infos[eval_i][0]['won'] muss True/False liefern.
    # Das kommt aus deinem BAR-Env-Wrapper.
    # ─────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_battles_won = 0
        eval_episode = 0
        eval_episode_rewards = []
        one_episode_rewards = []

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()

        # GRU-Startzustand für Evaluation: alles Nullen
        eval_rnn_states = np.zeros(
            (self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size),
            dtype=np.float32)
        eval_masks = np.ones(
            (self.n_eval_rollout_threads, self.num_agents, 1),
            dtype=np.float32)

        while True:
            self.trainer.prep_rollout()

            # Deterministisch: nimm die wahrscheinlichste Aktion, sample nicht
            eval_actions, eval_rnn_states = \
                self.trainer.policy.act(
                    np.concatenate(eval_obs),
                    np.concatenate(eval_rnn_states),
                    np.concatenate(eval_masks),
                    np.concatenate(eval_available_actions),
                    deterministic=True)

            eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))

            (eval_obs, eval_share_obs, eval_rewards, eval_dones,
             eval_infos, eval_available_actions) = self.eval_envs.step(eval_actions)

            one_episode_rewards.append(eval_rewards)
            eval_dones_env = np.all(eval_dones, axis=1)

            # GRU-Zustand bei Episode-Ende nullen
            eval_rnn_states[eval_dones_env == True] = np.zeros(
                ((eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size),
                dtype=np.float32)

            eval_masks = np.ones(
                (self.all_args.n_eval_rollout_threads, self.num_agents, 1),
                dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(
                ((eval_dones_env == True).sum(), self.num_agents, 1),
                dtype=np.float32)

            # Episode-Ende auswerten
            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    eval_episode_rewards.append(np.sum(one_episode_rewards, axis=0))
                    one_episode_rewards = []
                    # ─────────────────────────────────────────────────────────
                    # ANPASSUNG FÜR BAR:
                    # eval_infos[eval_i][0]['won'] → True wenn BAR-Match gewonnen
                    # Muss vom BAR-Env-Wrapper geliefert werden
                    # ─────────────────────────────────────────────────────────
                    if eval_infos[eval_i][0]['won']:
                        eval_battles_won += 1

            # Genug Episoden evaluiert → Ergebnisse loggen und beenden
            if eval_episode >= self.all_args.eval_episodes:
                eval_episode_rewards = np.array(eval_episode_rewards)
                eval_env_infos = {'eval_average_episode_rewards': eval_episode_rewards}
                self.log_env(eval_env_infos, total_num_steps)

                eval_win_rate = eval_battles_won / eval_episode
                print("eval win rate is {}.".format(eval_win_rate))

                if self.use_wandb:
                    wandb.log({"eval_win_rate": eval_win_rate}, step=total_num_steps)
                else:
                    self.writter.add_scalars("eval_win_rate",
                                             {"eval_win_rate": eval_win_rate},
                                             total_num_steps)
                break