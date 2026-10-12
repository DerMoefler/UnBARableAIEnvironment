from dataclasses import dataclass
import logging
import os
from pathlib import Path
from pprint import pformat
import signal
from src.environment.grpc_server import UnBARableAIGRPCServer
import subprocess
import threading
from typing import Optional, Dict, Any, List, Union, Callable

@dataclass(frozen=True)
class EngineSessionConfig:
    """
    All configuration parameters for an EngineSession,
    stdout is output stream for the engine's output, can be None (Terminal output), a file path, or subprocess.PIPE
    stderr is output stream for the engine's error output, can be None (Terminal output), a file path, or subprocess.PIPE
    """
    repo_root: Path = Path(__file__).resolve().parents[3]
    engine_root: Path = repo_root / "RecoilEngine"
    write_dir: Path = Path("~/bar-data/").expanduser()

    engine_exe: Path = engine_root / "build-amd64-linux" / "install" / "spring-headless"
    config_file: Path = engine_root / "config" / "test_fast.cfg"
    startscript: Path = engine_root / "startscripts" / "3PawnVs3Pawn.txt"

    stdout: Optional[Union[int, str, Path]] = Path("~/bar-data/engine_stdout.log").expanduser()
    stderr: Optional[Union[int, str, Path]] = Path("~/bar-data/engine_stderr.log").expanduser()
    merge_stderr_to_stdout: bool = False

    def __pos_init__(self):
        self.validate()

    def validate(self) -> None:
        if not self.repo_root.is_dir():
            raise ValueError(f"Repository root does not exist: {self.repo_root}")
        if not self.write_dir.is_dir():
            raise ValueError(f"Write directory does not exist: {self.write_dir}")
        if not self.engine_exe.is_file():
            raise ValueError(f"Engine xecutable does not exist: {self.engine_exe}")
        if not os.access(self.engine_exe, os.X_OK):
            raise PermissionError(f"spring-headless is not executable: {self.engine_exe}")


class EngineSession:
    def __init__(self, cfg: EngineSessionConfig, on_exit: Optional[Callable[[int], None]] = None):
        self.cfg: EngineSessionConfig = cfg

        self.on_exit = on_exit

        self.proc: Optional[subprocess.Popen] = None
        self._stdout_handle: Optional[Any] = None
        self._stderr_handle: Optional[Any] = None

        self._monitor_thread: Optional[threading.Thread] = None
        self.grpc_server = UnBARableAIGRPCServer()
        self.grpc_server.start()



    def _build_cmd(self) -> List[str]:
        """
        builds the command line to start the engine process, based on the configuration parameters.

        Parameters
        ----------
        None

        Returns
        -------
        List[str]:
            the command line as a list of strings, suitable for passing to subprocess.Popen
        """
        return [
            str(self.cfg.engine_exe),
            "--write-dir", str(self.cfg.write_dir),
            "--config", str(self.cfg.config_file),
            str(self.cfg.startscript),
        ]

    def _resolve_stream(self, stream_spec: Optional[Union[int, str, Path]]):
        """
        Resolves a stream specification to an actual stream object or None.

        Parameters
        ----------
        stream_spec : None | int | str | Path
            The stream specification. Can be:
            - None: Use the default stream (Terminal output).
            - int: Use the specified file descriptor (e.g., subprocess.PIPE).
            - str or Path: Use the specified file path. The file will be created if it does not exist.
        
        Returns
        -------
        None | int | BinaryIO
        The resolved stream that can be passed to subprocess.Popen.
        """
        if stream_spec is None:
            return None
        if stream_spec == subprocess.PIPE:
            return subprocess.PIPE
        if isinstance(stream_spec, (str, Path)):
            p = Path(stream_spec).expanduser()
            p.parent.mkdir(parents=True, exist_ok=True)
            return open(p, "wb")
        return stream_spec

    def start(self):
        """
        Starts the engine process if it is not already running.

        Returns
        -------
        info : Dict[str, Any]
            Information about the engine process, including PID, command line, working directory, write directory,
            running status, and exit code.
        """
        if self.is_running():
            return self.info()

        cmd = self._build_cmd()

        stdout = self._resolve_stream(self.cfg.stdout)
        stderr = self._resolve_stream(self.cfg.stderr)

        self._stdout_handle = stdout if hasattr(stdout, "close") else None
        self._stderr_handle = stderr if hasattr(stderr, "close") else None

        if self.cfg.merge_stderr_to_stdout:
            stderr = subprocess.STDOUT

        self.proc = subprocess.Popen(
            cmd,
            cwd=str(self.cfg.engine_root),
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
            text=True if (stdout == subprocess.PIPE or stderr == subprocess.PIPE) else False,
            bufsize=1,
        )

        self._monitor_thread = threading.Thread(
            target=self._monitor_process,
            args=(self.proc,),
            name="engine-process-monitor",
            daemon=True,
        )

        self._monitor_thread.start()

        logging.info("Engine process started: %s", pformat(self.info()))
        return self.info()

    def stop(self) -> None:
        """
        Beendet die Engine und schließt ihre Ausgabestreams.
        """

        try:
            if self.proc is None:
                return

            if self.proc.poll() is not None:
                self.proc.wait()
                logging.info("Engine process already exited ")
                return

            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass

            try:
                self.proc.wait(timeout=5)
                logging.info("Engine process terminated gracefully (in EngineSession.stop())")
            except subprocess.TimeoutExpired:
                logging.warning("Engine process did not terminate gracefully, forcing termination...")
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass

                self.proc.wait()

        finally:
            self.proc = None
            self._close_streams()
            self.grpc_server.stop()
            logging.info("Engine Session stop called, process terminated and streams closed.")

    def _monitor_process(self, proc: subprocess.Popen) -> None:
        """
        Wartet auf das Engine-Ende und ruft den Exit-Handler auf.

        Parameters
        ----------
        proc : subprocess.Popen
            The engine process to monitor.
        """
        return_code = proc.wait()

        logging.info("Engine process exited with return code: %s" " (0 = regular exit, -15 = SIGTERM, -9 = SIGKILL)", return_code)

        if self.on_exit is not None:
            self.on_exit(return_code)



    def _close_streams(self):
        """
        closes the stdout and stderr streams if they were opened by this session.
        """
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
        """
        Checks if the engine process is currently running.

        Returns
        -------
        bool:
            True if the engine process is running, False otherwise.
        """
        return self.proc is not None and self.proc.poll() is None

    def exit_code(self) -> Optional[int]:
        """
        Returns the exit code of the engine process if it has terminated, or None if it is still running.

        Returns
        -------
        Optional[int]:
            The exit code of the engine process, or None if it is still running.
        """
        if self.proc is None:
            return None
        return self.proc.poll()

    def info(self) -> Dict[str, Any]:
        """
        Returns information about the engine process
        
        Returns
        -------
        Dict[str, Any]:
            Information about the engine process, including PID, command line, working directory, write directory,
            running status, and exit code.
        """
        return {
            "pid": None if self.proc is None else self.proc.pid,
            "cmd": self._build_cmd(),
            "cwd": str(self.cfg.engine_root),
            "write_dir": str(self.cfg.write_dir),
            "log_file": self.cfg.stdout,
            "running": self.is_running(),
            "exit_code": self.exit_code(),
        }
