"""Random action agent for Metal Slug 6."""

from src.agents.base_agent import BaseAgent


class RandomAgent(BaseAgent):
    """Agent that takes random actions."""

    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, obs):
        return self.action_space.sample()
