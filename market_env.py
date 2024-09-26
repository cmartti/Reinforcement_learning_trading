import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env
from stable_baselines3 import PPO
import numpy as np
from market import Stock
from market import Bond
from gym import Env


class MarketEnv(gym.Env):
    def __init__(self, starting_balance=1000, trading_days=40, interest_rate=0.03, sigma=0.3, s0=100, b0=100):
        super(MarketEnv, self).__init__()
        self.reset_days = trading_days
        self.iterations = trading_days
        self.dt = trading_days / 252
        self.r = interest_rate  # Risk-free
        self.sigma = sigma
        self.mu = interest_rate
        self.s0 = s0
        self.balance = starting_balance
        self.b0 = b0
        self.action_space = spaces.Discrete(5)
        self.stock = Stock(self.mu, self.dt, self.s0, self.sigma)
        self.bond = Bond(self.b0, self.dt, self.r)

        # Convert bond_holdings and stock_holdings to Box instead of Discrete
        self.observation_space = spaces.Dict({
            'bond_price': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            'stock_price': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            'interest_rate': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),  # assuming interest rate between 0 and 1
            'balance': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),

            # Use Box spaces for holdings instead of Discrete to prevent one-hot encoding issues
            'bond_holdings': spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),  # integer holdings in float form
            'stock_holdings': spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32)   # integer holdings in float form
        })

        # Initial state
        self.state = {
            'bond_price': np.array([self.bond.get_price()], dtype=np.float32),
            'stock_price': np.array([self.stock.get_price()], dtype=np.float32),
            'interest_rate': np.array([self.r], dtype=np.float32),
            'balance': np.array([self.balance], dtype=np.float32),
            'bond_holdings': np.array([0], dtype=np.float32),
            'stock_holdings': np.array([0], dtype=np.float32)
        }

    def step(self, action):
        # Current price for stock and bond
        stock_price = self.stock.get_price()
        bond_price = self.bond.get_price()

        # Buy bond
        if action == 0:
            if self.state['balance'][0] >= bond_price and self.state['bond_holdings'][0] < 100:
                self.state['bond_holdings'][0] += 1
                self.state['balance'][0] -= bond_price
        # Sell bond
        elif action == 1:
            if self.state['bond_holdings'][0] > 0:
                self.state['bond_holdings'][0] -= 1
                self.state['balance'][0] += bond_price
        # Buy stock
        elif action == 2:
            if self.state['balance'][0] >= stock_price and self.state['stock_holdings'][0] < 100:
                self.state['stock_holdings'][0] += 1
                self.state['balance'][0] -= stock_price
        # Sell stock
        elif action == 3:
            if self.state['stock_holdings'][0] > 0:
                self.state['stock_holdings'][0] -= 1
                self.state['balance'][0] += stock_price
        # Do nothing
        elif action == 4:
            pass

        # Reward calculation
        previous_value = (
            self.state['balance'][0] +
            self.state['stock_holdings'][0] * self.state['stock_price'][0] +
            self.state['bond_holdings'][0] * self.state['bond_price'][0]
        )

        # Update prices for the next iteration
        self.stock.calculate_next_price()
        self.bond.calculate_next_price()
        self.state['stock_price'][0] = self.stock.get_price()
        self.state['bond_price'][0] = self.bond.get_price()

        # Reward is the total portfolio value change
        stock_value = self.state['stock_holdings'][0] * self.state['stock_price'][0]
        bond_value = self.state['bond_holdings'][0] * self.state['bond_price'][0]
        current_value = self.state['balance'][0] + stock_value + bond_value
        reward = current_value - previous_value
        reward = float(reward)
        
        # One less trading day
        self.iterations -= 1
        done = self.iterations <= 0  # Allow for a strict inequality to accommodate potential edge cases
        truncated = False

        return self.state, reward, done, truncated, {}

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed)
        self.stock.reset()
        self.bond.reset()
        self.iterations = self.reset_days  # Reset iterations to initial value
        self.state = {
            'bond_price': np.array([self.bond.get_price()], dtype=np.float32),
            'stock_price': np.array([self.stock.get_price()], dtype=np.float32),
            'interest_rate': np.array([self.r], dtype=np.float32),
            'balance': np.array([self.balance], dtype=np.float32),
            'bond_holdings': np.array([0], dtype=np.float32),
            'stock_holdings': np.array([0], dtype=np.float32)
        }
        return self.state, {}


# Instantiate the environment
env = MarketEnv()
env.reset(seed=42)

# Verifying that the env follows the Gym interface
check_env(env, warn=True, skip_render_check=True)

# Create the PPO agent with the environment
model = PPO("MultiInputPolicy", env, verbose=1, learning_rate=0.001)

# Train the agent for n timesteps 
model.learn(total_timesteps=20000)

# Save the trained agent
model.save("ppo_financial_model")

# Load the model if needed
# model = PPO.load("ppo_financial_model")

# Test the trained agent
obs, info = env.reset()
cur_reward = 0
rewards = []
for i in range(2000):
    #Range should be divisible by the amount of trading days
    #trading days are set in the init function.
    #Will be 2000/trading_days iterations
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    cur_reward += reward
    if done:
        rewards.append(cur_reward)
        obs, info = env.reset()  # Reset environment after an episode is done
        print(cur_reward)
        cur_reward = 0
print(rewards)
print(len(rewards))