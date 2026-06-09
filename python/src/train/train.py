"""
Training script integrating BAR environment, policy, replay buffer, and R_MAPPO trainer.

Example usage:
    python train.py --num-episodes 100 --num-mini-batch 4
"""

import argparse
import numpy as np
import torch
from bar_environment import BAR_Environment, EngineSessionConfig
from policy import R_MAPPO_Policy
from replay_buffer import SharedReplayBuffer
from r_mappo import R_MAPPO


class Args:
    """Config namespace for R_MAPPO trainer"""
    def __init__(self):
        self.clip_param = 0.2
        self.ppo_epoch = 10
        self.num_mini_batch = 4
        self.data_chunk_length = 4
        self.value_loss_coef = 1.0
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5
        self.huber_delta = 10.0
        
        self.use_recurrent_policy = False
        self.use_naive_recurrent_policy = False
        self.use_max_grad_norm = True
        self.use_clipped_value_loss = True
        self.use_huber_loss = False
        self.use_popart = False
        self.use_valuenorm = False
        self.use_value_active_masks = True
        self.use_policy_active_masks = True


def main():
    parser = argparse.ArgumentParser(description="Train BAR environment with R_MAPPO")
    parser.add_argument("--num-episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument("--num-mini-batch", type=int, default=4, help="Number of mini-batches")
    parser.add_argument("--buffer-size", type=int, default=256, help="Replay buffer size")
    parser.add_argument("--num-agents", type=int, default=2, help="Number of agents")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    
    args = parser.parse_args()
    device = torch.device(args.device)
    
    print("Initializing BAR environment...")
    env_config = EngineSessionConfig()
    env = BAR_Environment(env_config)
    
    # Placeholder dimensions (you'll need to adjust based on your actual obs/action spaces)
    obs_dim = 128  # TODO: Set to your actual observation dimension
    action_dim = 7  # TODO: Set to your actual action dimension
    
    print("Initializing policy...")
    policy = R_MAPPO_Policy(obs_dim, action_dim, device=device, lr=args.lr)
    
    print("Initializing replay buffer...")
    buffer = SharedReplayBuffer(
        num_agents=args.num_agents,
        obs_shape=(obs_dim,),
        action_shape=(1,),
        buffer_size=args.buffer_size,
        device=device
    )
    
    print("Initializing R_MAPPO trainer...")
    trainer_args = Args()
    trainer_args.num_mini_batch = args.num_mini_batch
    trainer = R_MAPPO(trainer_args, policy, device=device)
    
    print("\nStarting training loop...")
    for episode in range(args.num_episodes):
        # Reset environment
        obs, info = env.reset()
        
        # Convert observation to proper format
        if obs is None:
            obs = np.zeros((args.num_agents, obs_dim), dtype=np.float32)
        else:
            obs = np.array(obs, dtype=np.float32).reshape(args.num_agents, -1)
        
        share_obs = obs.mean(axis=0)  # Shared observation (centralized critic)
        
        episode_reward = 0.0
        done = False
        step_count = 0
        
        # Rollout phase
        trainer.prep_rollout()
        while not done and step_count < args.buffer_size:
            # Get actions from policy
            obs_tensor = torch.FloatTensor(obs).to(device)
            
            with torch.no_grad():
                # Simplified action selection (you'll need to adapt to your action space)
                action_logits = policy.actor(obs_tensor)
                action_dist = torch.distributions.Categorical(logits=action_logits)
                actions = action_dist.sample()
                action_log_probs = action_dist.log_prob(actions)
                
                # Get value predictions
                share_obs_tensor = torch.FloatTensor(share_obs).unsqueeze(0).to(device)
                values = policy.critic(share_obs_tensor)
            
            # Step environment
            step_result = env.step(actions.cpu().numpy())
            obs, reward, terminated, truncated, step_info = step_result
            
            done = terminated or truncated
            
            # Convert observation
            if obs is None:
                obs = np.zeros((args.num_agents, obs_dim), dtype=np.float32)
            else:
                obs = np.array(obs, dtype=np.float32).reshape(args.num_agents, -1)
            
            share_obs = obs.mean(axis=0)
            
            reward = np.array(reward, dtype=np.float32).reshape(args.num_agents, 1)
            
            # Store in buffer
            buffer.insert(
                share_obs=share_obs,
                obs=obs,
                actions=actions.cpu().numpy().reshape(args.num_agents, 1),
                action_log_probs=action_log_probs.cpu().numpy().reshape(args.num_agents, 1),
                value_preds=values.detach().cpu().numpy().reshape(1, 1),
                rewards=reward,
                masks=np.ones((args.num_agents, 1)) * (not done),
                active_masks=np.ones((args.num_agents, 1)),
            )
            
            episode_reward += reward.mean()
            step_count += 1
        
        # Compute returns and advantages
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).to(device)
            share_obs_tensor = torch.FloatTensor(share_obs).unsqueeze(0).to(device)
            next_value = policy.critic(share_obs_tensor).cpu().numpy()
        
        buffer.compute_returns(next_value, gamma=args.gamma)
        
        # Training phase
        trainer.prep_training()
        train_info = trainer.train(buffer, update_actor=True)
        
        # Reset buffer for next episode
        buffer.reset()
        
        # Logging
        print(f"Episode {episode + 1}/{args.num_episodes}")
        print(f"  Episode Reward: {episode_reward:.4f}")
        print(f"  Value Loss: {train_info.get('value_loss', 0):.6f}")
        print(f"  Policy Loss: {train_info.get('policy_loss', 0):.6f}")
        print(f"  Entropy: {train_info.get('dist_entropy', 0):.6f}")
        print()
    
    # Cleanup
    env.close()
    print("Training complete!")


if __name__ == "__main__":
    main()
