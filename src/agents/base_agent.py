"""Base agent interface for Metal Slug 6."""


class BaseAgent:
    """Base class for Metal Slug agents."""

    def predict(self, obs) -> int:
        raise NotImplementedError

    def learn(self, env, **kwargs):
        raise NotImplementedError("This agent does not support training")
