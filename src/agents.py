"""
agents.py — Optimized Mesa CustomerAgent with continuous emotional decay.
"""

import numpy as np
from mesa import Agent, Model

from src import config

class ServiceCenterABM(Model):
    """Mesa Model managing all CustomerAgents."""
    def __init__(self):
        super().__init__()
        self._next_id = 1

    def create_agent(self) -> "CustomerAgent":
        agent = CustomerAgent(self._next_id, self)
        self._next_id += 1
        return agent

    def step(self):
        pass


class CustomerAgent(Agent):
    """
    Customer Agent utilizing continuous emotional state [0.0 - 1.0].
    Emotion decays exponentially based on elapsed wait time.
    """
    def __init__(self, unique_id: int, model: ServiceCenterABM):
        super().__init__(model)
        self.unique_id = unique_id

        self.personality = np.random.choice(
            ["Conservative", "Steady", "Aggressive"],
            p=config.PERSONALITY_PROBS,
        )
        
        # Continuous emotion: 1.0 = ecstatic, 0.0 = completely frustrated
        self.emotion_val = np.clip(np.random.normal(config.EMOTION_INITIAL_MU, config.EMOTION_INITIAL_SIGMA), 0.1, 1.0)
        
        # Base patience drawn from lognormal distribution based on personality
        mu, sig = config.PATIENCE_PARAMS[self.personality]
        self.base_patience = np.random.lognormal(mu, sig)
        
        self.balk_threshold = config.BALK_THRESHOLDS[self.personality]

    @property
    def patience_threshold(self) -> float:
        """Effective patience is a function of base patience scaled by current emotion."""
        return max(2.0, self.base_patience * self.emotion_val)

    def decide_balk(self, queue_length: int) -> bool:
        """Balking likelihood increases as initial emotion drops."""
        threshold = self.balk_threshold
        if self.emotion_val < 0.5:
            threshold = max(1, int(threshold * 0.6))
        return queue_length >= threshold

    def update_emotion_from_wait(self, elapsed_wait: float):
        """Exponential emotional decay based on wait duration."""
        # Emotion decays slightly for every minute waited.
        # Decay rate is faster for Aggressive personalities.
        decay_rate = 0.005 if self.personality == "Conservative" else (0.01 if self.personality == "Steady" else 0.02)
        self.emotion_val *= np.exp(-decay_rate * elapsed_wait)

    @property
    def emotion_label(self) -> str:
        if self.emotion_val >= 0.7: return "Positive"
        if self.emotion_val >= 0.4: return "Neutral"
        return "Negative"

    def step(self):
        pass
