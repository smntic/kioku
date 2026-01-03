"""
Solve cartpole with PPO.
"""

from schedulers import StaticScheduler
from environments import GymEnvironment
from agents import PPOAgent
from trainers import Trainer
from utils import set_seed

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
print(results)
