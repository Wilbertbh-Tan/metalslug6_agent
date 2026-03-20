"""PPO agent wrapping a trained SB3 model for Metal Slug 6."""

from src.agents.base_agent import BaseAgent


class PPOAgent(BaseAgent):
    """Wraps a trained SB3 PPO model for inference."""

    def __init__(self, model_path: str, deterministic: bool = True):
        from stable_baselines3 import PPO

        self.model = PPO.load(model_path)
        self.deterministic = deterministic

    def predict(self, obs):
        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        return action

    def learn(self, env, **kwargs):
        """Train the PPO model — delegates to SB3."""
        self.model.learn(**kwargs)
