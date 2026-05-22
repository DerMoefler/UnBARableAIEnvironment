from bar_environment import BAR_Environment
import time


env = BAR_Environment(
    stdout_log="/home/dorian/bar-data/spring_stdout.log",
    stderr_log="/home/dorian/bar-data/spring_stderr.log",
)

obs, info = env.reset()
print(info)  # pid, cmd, cwd, write_dir

# Give the process a moment to start and emit logs
time.sleep(1000)

# ... später:
env.close()