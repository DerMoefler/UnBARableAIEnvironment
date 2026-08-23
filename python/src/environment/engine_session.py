import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Callable

from src.environment.shared_memory_reader import SharedMemoryReader

@dataclass
class EngineSessionConfig:
    """
    all configuration parameters for an EngineSession,
    stdout is output stream for the engine's output, can be None (Terminal output), a file path, or subprocess.PIPE
    stderr is output stream for the engine's error output, can be None (Terminal output), a file path, or subprocess.PIPE
    """
    engine_exe: Union[str, Path] = "~/repos/UnBARableAIEnvironment/RecoilEngine/build-amd64-linux/install/spring-headless"
    write_dir: Union[str, Path] = "~/bar-data"
    config_file: Union[str, Path] = "config/test_fast.cfg"
    startscript: Union[str, Path] = "startscripts/3PawnVs3Pawn.txt"
    cwd: Union[str, Path] = "~/repos/UnBARableAIEnvironment/RecoilEngine"

    stdout: Optional[Union[int, str, Path]] = "~/bar-data/engine_stdout.log"
    stderr: Optional[Union[int, str, Path]] = "~/bar-data/engine_stderr.log"
    merge_stderr_to_stdout: bool = False
    text_mode: bool = True

    shared_memory_reader_factory: Optional[Callable[[], Any]] = None
    ipc_ready_timeout_s: float = 10.0
    ipc_poll_interval_s: float = 0.05


class EngineSession:
    def __init__(self, cfg: EngineSessionConfig):
        self.cfg = cfg

        self.engine_exe = Path(cfg.engine_exe).expanduser()
        self.write_dir = Path(cfg.write_dir).expanduser()
        self.config_file = Path(cfg.config_file).expanduser()
        self.startscript = Path(cfg.startscript).expanduser()
        self.cwd = Path(cfg.cwd).expanduser()

        self.proc: Optional[subprocess.Popen] = None
        self._stdout_handle: Optional[Any] = None
        self._stderr_handle: Optional[Any] = None
        self.reader = SharedMemoryReader(
            ipc_factory=cfg.shared_memory_reader_factory,
            ready_timeout_s=cfg.ipc_ready_timeout_s,
            poll_interval_s=cfg.ipc_poll_interval_s,
        )


    def _build_cmd(self) -> List[str]:
        return [
            str(self.engine_exe),
            "--write-dir", str(self.write_dir),
            "--config", str(self.config_file),
            str(self.startscript),
        ]

    def _resolve_stream(self, stream_spec):
        if stream_spec is None:
            return None
        if stream_spec == subprocess.PIPE:
            return subprocess.PIPE
        if isinstance(stream_spec, (str, Path)):
            p = Path(stream_spec).expanduser()
            p.parent.mkdir(parents=True, exist_ok=True)
            return open(p, "wb")
        return stream_spec

    def _connect_ipc(self) -> None:
        self.reader.connect()

    def _disconnect_ipc(self) -> None:
        self.reader.close()

    def start(self) -> Dict[str, Any]:
        if self.is_running():
            return self.info()

        if not self.engine_exe.is_file():
            raise FileNotFoundError(f"spring-headless not found: {self.engine_exe}")
        if not os.access(self.engine_exe, os.X_OK):
            raise PermissionError(f"spring-headless is not executable: {self.engine_exe}")
        if not self.cwd.exists():
            raise FileNotFoundError(f"cwd does not exist: {self.cwd}")

        self.write_dir.mkdir(parents=True, exist_ok=True)
        cmd = self._build_cmd()

        stdout = self._resolve_stream(self.cfg.stdout)
        stderr = self._resolve_stream(self.cfg.stderr)

        self._stdout_handle = stdout if hasattr(stdout, "close") else None
        self._stderr_handle = stderr if hasattr(stderr, "close") else None

        if self.cfg.merge_stderr_to_stdout:
            stderr = subprocess.STDOUT

        self.proc = subprocess.Popen(
            cmd,
            cwd=str(self.cwd),
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
            text=self.cfg.text_mode if (stdout == subprocess.PIPE or stderr == subprocess.PIPE) else False,
            bufsize=1 if self.cfg.text_mode else 0,
        )

        if self.cfg.shared_memory_reader_factory is not None:
            self._connect_ipc()
        return self.info()

    def stop(self) -> None:
        if self.proc is None:
            self._disconnect_ipc()
            self._close_streams()
            return

        if self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass

            for _ in range(50):
                if self.proc.poll() is not None:
                    break
                time.sleep(0.01)

            if self.proc.poll() is None:
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass

        self.proc = None
        self._disconnect_ipc()
        self._close_streams()

    def _close_streams(self):
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

    def get_world_frame(self) -> int:
        return self.reader.get_world_frame()

    def get_all_units(self):
        return self.reader.get_all_units()

    def get_alive_units(self):
        return self.reader.get_alive_units()