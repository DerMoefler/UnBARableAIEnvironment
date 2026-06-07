from bar_environment import BAR_Environment


def main():
	env = BAR_Environment()
	observation = env.get_obs_agent(agent_id=0)


if __name__ == "__main__":
	main()


