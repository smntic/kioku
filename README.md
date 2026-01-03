# Kioku

A cognitive science-inspired reinforcement learning research library. Kioku is
designed to be modular and give you tools to experiment with classical and
modern RL algorithms and build your own.

## Features

Kioku currently includes implementations of DQN, A2C and PPO, along with
reusable components for memory buffers, schedulers, models and environments. It
supports the following agents based on classical and SOTA RL methods:

- [**DQN**](https://github.com/smntic/kioku/blob/main/kioku/agents/dqn_agent.py)
  - Based on: [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)
  - Uses Q-value function with Polyak averaging and epsilon decay
- [**DQN with PER**](https://github.com/smntic/kioku/blob/main/kioku/agents/dqn_per_agent.py)
  - Adds a priotized experience replay buffer based on: [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)
- [**A2C**](https://github.com/smntic/kioku/blob/main/kioku/agents/a2c_agent.py)
  - Based on: [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)
  - Uses N-step returns
- [**PPO**](https://github.com/smntic/kioku/blob/main/kioku/agents/ppo_agent.py)
  - Based on: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
  - Uses Generalized Advantage Estimation, N-step returns, mini-batch learning,
    multiple learning iterations per batch.

## Installation

Install via PyPI:

```bash
pip install kioku-rl
```

## Usage

Run an example:

```bash
python examples/ppo_cartpole.py
```

or use it in your own programs as a library:

```python
from kioku.schedulers import StaticScheduler
from kioku.environments import GymEnvironment
from kioku.agents import PPOAgent
from kioku.trainers import Trainer
from kioku.utils import set_seed

set_seed(42)

env = GymEnvironment("CartPole-v1")
render_env = GymEnvironment("CartPole-v1", render_mode="human")

agent = PPOAgent(
    observation_size=env.observation_size,
    num_actions=env.action_size,
    learning_rate=StaticScheduler(1e-3)
)

trainer = Trainer(agent, env)
trainer.train(100000)

tester = Trainer(agent, render_env)
results = tester.test(10)
```

## Results

**DQN Cartpole:**

![DQN cartpole episde rewards](./results/dqn_cartpole/dqn_cartpole.png)
![DQN cartpole gif](./results/dqn_cartpole/dqn_cartpole.gif)

**PPO Cartpole:**

![PPO CartPole episode rewards](./results/ppo_cartpole/ppo_cartpole.png)
![PPO CartPole gif](results/ppo_cartpole/ppo_cartpole.gif)

**PPO Lunar Lander:**

![PPO Lunar Lander episode rewards](./results/ppo_lunarlander/ppo_lunarlander.png)
![PPO Lunar Lander gif](./results/ppo_lunarlander/ppo_lunarlander.gif)
