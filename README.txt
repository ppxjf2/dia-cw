The yaml file dia_cw.yaml should have all of the packages that are necessary for the project to work. 

The two files to run are genetic_algorithm_run.py which has code at the bottom to uncomment depending on what you want to do, and monte_carlo_run.py which also has some code to uncomment in it depending on what you want to do.

genetic_algorithm_run.py
	uncomment train() to train a new genetic algorithm
	uncomment bestAgentRun() to run the agent with the best score
	uncomment graph() for graphs of score for comparison

monte_carlo_run.py
	uncomment test.train(1000) to train a new Monte-Carlo algorithm
	uncomment example.run(env) for Monte-Carlo soft policy
	uncomment example.runPillCollect(env) for Monte-Carlo pill collecting behaviour
	uncomment example.runGhostEvade(env) for Monte-Carlo ghost evading behaviour