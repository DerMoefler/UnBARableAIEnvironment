import logging
import os
import re
import signal
import subprocess
import threading
import uuid

from dataclasses import dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Optional, Dict, Any, List, Callable

import bar_ai
from src.environment.grpc_server import UnBARableAIGRPCServer
from bar_ai import SharedMemory

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class EngineSessionConfig:
    """
    All configuration parameters for an EngineSession,
    stdout is output stream for the engine's output, can be None (Terminal output), a file path, or subprocess.PIPE
    stderr is output stream for the engine's error output, can be None (Terminal output), a file path, or subprocess.PIPE
    """
    engine_id: int
    repo_root: Path = Path(__file__).resolve().parents[3]
    engine_root: Path = repo_root / "RecoilEngine"
    isolation_dir: Path = Path("~/bar-data/").expanduser()
    write_dir: Path = field(init=False)

    engine_exe: Path = engine_root / "build-amd64-linux" / "install" / "spring-headless"
    config_file: Path = engine_root / "config" / "test_fast.cfg"
    startscript: Path = engine_root / "startscripts" / "3vs3PawnsNew.txt"

    # TODO change to write_dir/engine_stdout/err.log
    default_stdout: int | str | Path = Path("~/bar-data/engine_stdout.log").expanduser()
    default_stderr: int | str | Path = Path("~/bar-data/engine_stderr.log").expanduser()
    merge_stderr_to_stdout: bool = False

    def __post_init__(self):
        self.validate()
        object.__setattr__(self, "write_dir", self.isolation_dir / "instances" / f"engine-{self.engine_id:03d}")
        self.ensure_write_dir()

    def validate(self) -> None:
        if not self.repo_root.is_dir():
            raise ValueError(f"Repository root does not exist: {self.repo_root}")
        if not self.isolation_dir.is_dir():
            raise ValueError(f"Isolation directory does not exist: {self.isolation_dir}")
        if not self.engine_exe.is_file():
            raise ValueError(f"Engine executable does not exist: {self.engine_exe}")
        if not os.access(self.engine_exe, os.X_OK):
            raise PermissionError(f"Engine is not executable: {self.engine_exe}")

    def ensure_write_dir(self) -> None:
        """
        Checks that a directory isolation_dir / "instances" / "engine-{self.engine_id}" exists

        Creates if it doesn't exist yet.
        """
        instances_dir = self.write_dir.resolve().parents[0]
        if not instances_dir.is_dir():
            Path.mkdir(instances_dir)

        if not self.write_dir.is_dir():
            Path.mkdir(self.write_dir)

class EngineSession:
    def __init__(self, cfg: EngineSessionConfig, on_exit: Optional[Callable[[int], None]] = None):
        self.cfg: EngineSessionConfig = cfg

        self.on_exit = on_exit

        self.proc: subprocess.Popen | None = None

        self.stdout = self.cfg.default_stdout
        self.stderr = self.cfg.default_stderr
        self._stdout_handle: Optional[Any] = None
        self._stderr_handle: Optional[Any] = None

        self._monitor_thread: Optional[threading.Thread] = None
        self.grpc_server: UnBARableAIGRPCServer = UnBARableAIGRPCServer()
        self.shared_memory: SharedMemory | None = None

    def start(self):
        """
        Starts the engine process if it is not already running.

        After creating the engine process, it starts the gRPC-Server and creates the shared-memory.
        See _initialize_shm for an explanation on the shm creation.

        Returns
        -------
        info : Dict[str, Any]
            Information about the engine process, including PID, command line, working directory, write directory,
            running status, and exit code.
        """
        if self.is_running():
            return self.info()

        cmd = self._get_build_cmd()

        stdout = self._resolve_stream(self.stdout)
        stderr = self._resolve_stream(self.stderr)

        self._stdout_handle = stdout if hasattr(stdout, "close") else None
        self._stderr_handle = stderr if hasattr(stderr, "close") else None

        if self.cfg.merge_stderr_to_stdout:
            stderr = subprocess.STDOUT

        self.grpc_server.start()
        self._initialize_shm()

        assert self.shared_memory is not None
        # Create environment for engine
        env = os.environ.copy()
        env["UNBARABLEAI_SHM_NAME"] = self.shared_memory.get_name()

        self.proc = subprocess.Popen(
            cmd,
            cwd=str(self.cfg.engine_root),
            stdout=stdout,
            stderr=stderr,
            env=env,
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

        logger.info("Engine process started: %s", pformat(self.info()))
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
                logger.info("Engine process already exited ")
                return

            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass

            try:
                self.proc.wait(timeout=5)
                logger.info("Engine process terminated gracefully (in EngineSession.stop())")
            except subprocess.TimeoutExpired:
                logger.warning("Engine process did not terminate gracefully, forcing termination...")
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass

                self.proc.wait()

        finally:
            self.proc = None
            self._close_streams()
            self.grpc_server.stop()
            if self.shared_memory is not None:
                SharedMemory.remove(self.shared_memory.get_name())
                self.shared_memory = None
            logger.info("Engine Session stop called, process terminated and streams closed.")

    def is_running(self) -> bool:
        """
        Checks if the engine process is currently running.

        Returns
        -------
        bool:
            True if the engine process is running, False otherwise.
        """
        return self.proc is not None and self.proc.poll() is None

    def get_engine_id(self) -> int:
        return self.cfg.engine_id

    @property
    def process_id(self) -> int | None:
        return None if self.proc is None else self.proc.pid

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
            "cmd": self._get_build_cmd(),
            "cwd": str(self.cfg.engine_root),
            "isolation_dir": str(self.cfg.isolation_dir),
            "write_dir": str(self.cfg.write_dir),
            "log_file": self.stdout,
            "running": self.is_running(),
            "exit_code": self.exit_code(),
        }

    def _initialize_shm(self) -> None:
        """
        Initializes the shared-memory.
        """
        assert self.grpc_server.port is not None

        shared_memory_name = f"/bar_ai_{uuid.uuid4().hex}"

        startscript: Path = self.cfg.startscript
        if not startscript.is_file():
            raise ValueError("Startscript file does not exist")

        # Simple regex for scheme "{num_ally_units}vs{num_enemy_units}"
        # vs is case insensitive and the number has dynamic extent, i.e. 1 but also 100 works
        match = re.match(r"^(\d+)vs(\d+)", startscript.stem, re.IGNORECASE)
        if not match:
            raise ValueError(
                f"Startscript name must start with '<numAllyUnits>vs<numEnemyUnits>', got"
                "'{startscript.name}'")

        num_ally_units = int(match.group(1))
        num_enemy_units = int(match.group(2))


        # TODO change status if enum supports state better
        self.shared_memory = SharedMemory.create(
            name=shared_memory_name,
            status=bar_ai.EngineStatus.RUNNING,
            connection=bar_ai.GrpcConnection(self.grpc_server.port),
            initial_num_ally_units=num_ally_units,
            initial_num_enemy_units=num_enemy_units,
        )


    def _monitor_process(self, proc: subprocess.Popen) -> None:
        """
        Wartet auf das Engine-Ende und ruft den Exit-Handler auf.

        Parameters
        ----------
        proc : subprocess.Popen
            The engine process to monitor.
        """
        return_code = proc.wait()

        logger.info("Engine process exited with return code: %s" " (0 = regular exit, -15 = SIGTERM, -9 = SIGKILL)", return_code)

        if self.on_exit is not None:
            self.on_exit(return_code)

    def _get_build_cmd(self) -> List[str]:
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
        # TODO: The config seems to be opened as writeable but is shared across engine's when used within a pool.
        # Maybe required to copy to instance_dir or something.
        return [
            str(self.cfg.engine_exe),
            "--isolation-dir", str(self.cfg.isolation_dir),
            "--write-dir", str(self.cfg.write_dir),
            "--config", str(self.cfg.config_file),
            str(self.cfg.startscript),
        ]


    def _resolve_stream(self, stream_spec: int | str | Path):
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
        if stream_spec == subprocess.PIPE:
            return subprocess.PIPE
        if isinstance(stream_spec, (str, Path)):
            p = Path(stream_spec).expanduser()
            p.parent.mkdir(parents=True, exist_ok=True)
            return open(p, "wb")
        return stream_spec

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
