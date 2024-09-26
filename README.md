This is a Reinforcement Learning model that trains a agent to be able to optimally pick a portfolio. 
The agent can at each time period (trading day) choose to buy/sell either a stock or bond.
The agent can also choose to hold its current portfolio. 

The model is trained in an environment created by myself and for each iteration in the environment,
the agent gets a reward based on the return of its portfolio between the current time-point and the
previous time-point.

Currently, the next periods stock price is simulated using a GBM with constant variance. The price of the
bond is also calculated in a simple way with only increasing the price with the current periods EAR, both
of these methods will be expanded in the future and other assets will also be added to the market.

