A Reinforcement Learning model that trains a agent to be able to optimally pick a portfolio. 
The agent can at each time period (trading day) choose to buy/sell either a stock or bond.
The agent can also choose to hold its current portfolio. Currently, the next periods stock price
is simulated using a GBM.

The model is trained in an environment created by myself and for each iteration in the environment,
the agent gets a reward based on the return of its portfolio between the current time-point and the
previous time-point.
