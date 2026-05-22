import os
import signal
import subprocess
from pathlib import Path


class BAR_Environment:
    def __init__(
        self,
        engine_exe="~/RecoilEngine/build-amd64-linux/install/spring-headless",
        write_dir="/home/dorian/bar-data",
        config="~/RecoilEngine/config/test_fast.cfg",
        startscript="~/RecoilEngine/startscripts/3PawnVs3Pawn.txt",
        cwd="~/RecoilEngine",
        stdout_log=None,
        stderr_log=None,
    ):
        """
        Minimaler Environment-Skeleton der beim reset() einen spring-headless Prozess startet.

        Parameter:
          - engine_exe: Pfad zum spring-headless Binary (relativ oder absolut)
          - write_dir: --write-dir Ziel
          - config: --config Pfad (kann ~ enthalten)
          - startscript: Startscript (kann relativ zu cwd sein)
          - cwd: Arbeitsverzeichnis für den Prozess (wichtig für relative Pfade)
          - stdout_log / stderr_log: optional Log-Dateien
        """
        self.engine_exe = Path(engine_exe).expanduser()
        self.write_dir = Path(write_dir).expanduser()
        self.config = Path(config).expanduser()
        self.startscript = Path(startscript).expanduser()
        self.cwd = Path(cwd).expanduser()

        self.proc = None  # subprocess.Popen handle

        self.stdout_log = Path(stdout_log).expanduser() if stdout_log else None
        self.stderr_log = Path(stderr_log).expanduser() if stderr_log else None

        # Keep opened filehandles here so we can close them on terminate
        self.stdout_handle = None
        self.stderr_handle = None

    def _terminate_proc(self):
        """Beendet einen laufenden spring-headless Prozess (falls vorhanden) sauber."""
        if self.proc is None:
            return

        if self.proc.poll() is None:
            # Prozess läuft noch -> erst SIGTERM an Prozessgruppe, dann ggf. SIGKILL
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass

            # Kurze, nicht-blockierende "Soft"-Wartephase ohne Timeouts zu versprechen:
            # Wir prüfen nur ein paar mal, ob er schon tot ist.
            for _ in range(50):
                if self.proc.poll() is not None:
                    break

            if self.proc.poll() is None:
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass

        self.proc = None

        # Close any filehandles we opened for logging
        try:
            if getattr(self, "stdout_handle", None):
                try:
                    self.stdout_handle.close()
                except Exception:
                    pass
                self.stdout_handle = None
            if getattr(self, "stderr_handle", None):
                try:
                    self.stderr_handle.close()
                except Exception:
                    pass
                self.stderr_handle = None
        except Exception:
            pass

    def reset(self, *args, **kwargs):
        """
        Reset: beendet ggf. laufenden Prozess und startet einen neuen spring-headless Run.

        Rückgabe:
          observation, info  (aktuell noch Platzhalter)
        """
        # 1) Alten Prozess beenden
        self._terminate_proc()

        # 2) Vorbedingungen prüfen
        if not self.engine_exe.is_file():
            raise FileNotFoundError(f"spring-headless nicht gefunden: {self.engine_exe}")

        # Optional: wenn nicht executable, kann man chmod +x brauchen
        if not os.access(self.engine_exe, os.X_OK):
            raise PermissionError(f"spring-headless ist nicht ausführbar: {self.engine_exe}")

        # write_dir anlegen, falls nicht vorhanden
        self.write_dir.mkdir(parents=True, exist_ok=True)

        # cwd prüfen
        if not self.cwd.exists():
            raise FileNotFoundError(f"cwd existiert nicht: {self.cwd}")

        # 3) Command bauen (ohne shell=True!)
        # Dein gewünschter Prozess:
        # RecoilEngine/build-amd64-linux/install/spring-headless --write-dir /home/dorian/bar-data
        #   --config ~/RecoilEngine/config/test_fast.cfg startscripts/3PawnVs3Pawn.txt
        cmd = [
            str(self.engine_exe),
            "--write-dir", str(self.write_dir),
            "--config", str(self.config),
            str(self.startscript),
        ]

        # 4) stdout/stderr handling
        # Open filehandles only if paths were provided; otherwise inherit parent's streams
        stdout_handle = open(self.stdout_log, "ab") if self.stdout_log else None
        stderr_handle = open(self.stderr_log, "ab") if self.stderr_log else None

        try:
            # start_new_session=True -> eigener Prozessgruppenleader (wichtig zum Killen der Gruppe)
            self.proc = subprocess.Popen(
                cmd,
                cwd=str(self.cwd),
                stdout=None,#stdout_handle,
                stderr=None, #stderr_handle,
                start_new_session=True,
                bufsize=0,
            )
        except Exception:
            # Falls wir Filehandles geöffnet haben, wieder schließen
            if stdout_handle and hasattr(stdout_handle, "close"):
                stdout_handle.close()
            if stderr_handle and hasattr(stderr_handle, "close"):
                stderr_handle.close()
            raise

        # keep handles for later cleanup
        self.stdout_handle = stdout_handle
        self.stderr_handle = stderr_handle

        info = {
            "pid": self.proc.pid,
            "cmd": cmd,
            "cwd": str(self.cwd),
            "write_dir": str(self.write_dir),
        }

        # TODO: observation aus Game-State/Log/IPC ableiten
        observation = None
        return observation, info

    def step(self, action):
        """
        Platzhalter: In Zukunft wird hier:
          - action -> Spring Commands / AI Interface / Lua Messages / etc.
          - observation/reward/termination aus dem Spiel auslesen
        """
        observation = None
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info

    def close(self):
        """Explizites Cleanup."""
        self._terminate_proc()

    def __del__(self):
        # Best effort cleanup, falls Objekt eingesammelt wird
        try:
            self.close()
        except Exception:
            pass