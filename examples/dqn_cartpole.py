"""
Solve cartpole with DQN.
"""

from kioku.schedulers import StaticScheduler
from kioku.environments import GymEnvironment
from kioku.agents import DQNAgent
from kioku.trainers import Trainer
from kioku.utils import set_seed

set_seed(42)

env = GymEnvironment("CartPole-v1")
render_env = GymEnvironment("CartPole-v1", render_mode="human")

agent = DQNAgent(
    observation_size=env.observation_size,
    num_actions=env.action_size,
    learning_rate=StaticScheduler(1e-2)
)

trainer = Trainer(agent, env)
trainer.train(10000)

tester = Trainer(agent, render_env)
results = tester.test(10)
print(results)
