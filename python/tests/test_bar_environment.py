from src.environment.bar_environment import BAR_Environment

def main():
	"""
	Runs the basic BAR environment observation smoke test.

	Returns
	-------
	None
	"""
	env = BAR_Environment()
	observation = env.get_obs_agent(agent_id=0)
	print(observation)


if __name__ == "__main__":
	main()


