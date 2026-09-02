import os
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path
import threading
import logging
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



class EngineSession:
    def __init__(self, cfg: EngineSessionConfig, on_exit: Optional[Callable[[], None]] = None):
        self.cfg = cfg
        self.on_exit = on_exit

        self.engine_exe = Path(cfg.engine_exe).expanduser()
        self.write_dir = Path(cfg.write_dir).expanduser()
        self.config_file = Path(cfg.config_file).expanduser()
        self.startscript = Path(cfg.startscript).expanduser()
        self.cwd = Path(cfg.cwd).expanduser()

        self.proc: Optional[subprocess.Popen] = None
        self._stdout_handle: Optional[Any] = None
        self._stderr_handle: Optional[Any] = None

        self.reader = SharedMemoryReader()

        self._monitor_thread: Optional[threading.Thread] = None



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
            str(self.engine_exe),
            "--write-dir", str(self.write_dir),
            "--config", str(self.config_file),
            str(self.startscript),
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

    def start(self) -> Dict[str, Any]:
        """
        Starts the engine process if it is not already running.

        Returns
        -------
        Dict[str, Any]:
            Information about the engine process, including PID, command line, working directory, write directory,
            running status, and exit code.
        """
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
                return

            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass

            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass

                self.proc.wait()

        finally:
            self.proc = None
            self._close_streams()
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
            self.on_exit()



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
            "cwd": str(self.cwd),
            "write_dir": str(self.write_dir),
            "running": self.is_running(),
            "exit_code": self.exit_code(),
        }
