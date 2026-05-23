import os
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List, Union


@dataclass
class EngineSessionConfig:
    engine_exe: Union[str, Path] = "~/RecoilEngine/build-amd64-linux/install/spring-headless"
    write_dir: Union[str, Path] = "~/bar-data"
    config_file: Union[str, Path] = "config/test_fast.cfg"
    startscript: Union[str, Path] = "startscripts/3PawnVs3Pawn.txt"
    cwd: Union[str, Path] = "~/RecoilEngine"

    # Output/Debug:
    # - None => erbt Terminal (falls engine überhaupt etwas auf stdout/stderr schreibt)
    # - subprocess.PIPE => du kannst später programmgesteuert lesen
    # - Dateipfad => schreibt Logs in Datei
    stdout: Optional[Union[int, str, Path]] = None
    stderr: Optional[Union[int, str, Path]] = None

    # Falls True: stderr wird in stdout gemerged (praktisch fürs Debug)
    merge_stderr_to_stdout: bool = False

    # Textmode fürs Pipe-Lesen
    text_mode: bool = True


class EngineSession:
    """
    Verwaltet genau eine Engine-Instanz (spring-headless Prozess).
    start() startet den Prozess, stop() beendet ihn.
    """

    def __init__(self, cfg: EngineSessionConfig):
        self.cfg = cfg

        self.engine_exe = Path(cfg.engine_exe).expanduser()
        self.write_dir = Path(cfg.write_dir).expanduser()
        self.config_file = Path(cfg.config_file).expanduser()
        self.startscript = Path(cfg.startscript).expanduser()
        self.cwd = Path(cfg.cwd).expanduser()

        self.proc: Optional[subprocess.Popen] = None
        self._stdout_handle = cfg.stdout
        self._stderr_handle = cfg.stderr

    def _build_cmd(self) -> List[str]:
        return [
            str(self.engine_exe),
            "--write-dir", str(self.write_dir),
            "--config", str(self.config_file),
            str(self.startscript),
        ]

    def _resolve_stream(self, stream_spec):
        """
        stream_spec:
          - None -> None (Terminal erben)
          - subprocess.PIPE -> PIPE
          - str/Path -> Datei-Handle (append binary)
        """
        if stream_spec is None:
            return None
        if stream_spec == subprocess.PIPE:
            return subprocess.PIPE
        if isinstance(stream_spec, (str, Path)):
            p = Path(stream_spec).expanduser()
            p.parent.mkdir(parents=True, exist_ok=True)
            return open(p, "ab")  # binary append
        return stream_spec

    def start(self) -> Dict[str, Any]:
        """
        Startet die Engine. Wirft Exceptions, wenn Pfade nicht stimmen oder Start fehlschlägt.
        Gibt ein info-dict zurück (pid, cmd, cwd, write_dir).
        """
        if self.is_running():
            # idempotent: wenn schon läuft, einfach info zurück
            return self.info()

        # Vorbedingungen
        if not self.engine_exe.is_file():
            raise FileNotFoundError(f"spring-headless nicht gefunden: {self.engine_exe}")
        if not os.access(self.engine_exe, os.X_OK):
            raise PermissionError(f"spring-headless ist nicht ausführbar: {self.engine_exe}")
        if not self.cwd.exists():
            raise FileNotFoundError(f"cwd existiert nicht: {self.cwd}")

        self.write_dir.mkdir(parents=True, exist_ok=True)
        cmd = self._build_cmd()

        # Streams konfigurieren
        stdout = self._resolve_stream(self.cfg.stdout)
        stderr = self._resolve_stream(self.cfg.stderr)

        # Handles merken, damit wir sie später schließen können (wenn Datei)
        self._stdout_handle = stdout if hasattr(stdout, "close") else None
        self._stderr_handle = stderr if hasattr(stderr, "close") else None

        if self.cfg.merge_stderr_to_stdout:
            # stderr in stdout mergen (falls stdout PIPE oder Datei/Terminal ist)
            stderr = subprocess.STDOUT

        self.proc = subprocess.Popen(
            cmd,
            cwd=str(self.cwd),
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,          # eigene Prozessgruppe -> sauberes Killen
            text=self.cfg.text_mode if (stdout == subprocess.PIPE or stderr == subprocess.PIPE) else False,
            bufsize=1 if self.cfg.text_mode else 0,
        )

        return self.info()

    def stop(self) -> None:
        """
        Beendet die EngineSession (falls laufend). Idempotent.
        """
        if self.proc is None:
            self._close_streams()
            return

        if self.proc.poll() is None:
            # Prozessgruppe beenden (inkl. children)
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass

            # ein paar Polls; falls noch lebt -> SIGKILL
            for _ in range(50):
                if self.proc.poll() is not None:
                    break

            if self.proc.poll() is None:
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass

        self.proc = None
        self._close_streams()

    def _close_streams(self):
        # Datei-Handles schließen, wenn wir welche geöffnet haben
        try:
            if self._stdout_handle:
                self._stdout_handle.close()
        finally:
            self._stdout_handle = None
        try:
            if self._stderr_handle and self._stderr_handle is not self._stdout_handle:
                self._stderr_handle.close()
        finally:
            self._stderr_handle = None

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def exit_code(self) -> Optional[int]:
        if self.proc is None:
            return None
        return self.proc.poll()

    def info(self) -> Dict[str, Any]:
        return {
            "pid": None if self.proc is None else self.proc.pid,
            "cmd": self._build_cmd(),
            "cwd": str(self.cwd),
            "write_dir": str(self.write_dir),
            "running": self.is_running(),
            "exit_code": self.exit_code(),
        }

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()
        return False