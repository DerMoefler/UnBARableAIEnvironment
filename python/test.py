from bar_environment import BAR_Environment
import time


env = BAR_Environment()

obs, info = env.reset()
print(info)  # pid, cmd, cwd, write_dir

# Give the process a moment to start and emit logs
time.sleep(1000)


# ... später:
env.close()