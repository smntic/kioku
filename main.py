"""
main.py

This module contains the main script that trains and tests the agent.

Ideally, the main script should be kept as simple as possible. It should
generally only create objects and run functions. Try not to place any complex
logic here and instead abstract it away.
"""

from schedulers import StaticScheduler
from environments import GymEnvironment
from agents import PPOAgent
from trainers import Trainer
import torch
import numpy as np

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

env = GymEnvironment("CartPole-v1", render_mode="human")
render_env = GymEnvironment("CartPole-v1")

agent = PPOAgent(
    observation_size=env.observation_size,
    num_actions=env.action_size,
)

trainer = Trainer(agent, env)
trainer.train(5000)

tester = Trainer(agent, render_env)
results = tester.test(10)
print(results)
