"""
agents.py — Optimized CustomerAgent with continuous emotional decay.
Independent of Mesa to avoid API mismatches and optimize speed.
"""

import numpy as np
from src import config

class ServiceCenterABM:
    """Model managing all CustomerAgents."""
    def __init__(self):
        self._next_id = 1

    def create_agent(self, rng_agent: np.random.RandomState) -> "CustomerAgent":
        agent = CustomerAgent(self._next_id, self, rng_agent)
        self._next_id += 1
        return agent

class CustomerAgent:
    """
    Customer Agent utilizing continuous emotional state [0.0 - 1.0].
    Emotion decays exponentially based on elapsed wait time.
    """
    def __init__(self, unique_id: int, model: ServiceCenterABM, rng_agent: np.random.RandomState):
        self.unique_id = unique_id
        self.model = model

        self.personality = rng_agent.choice(
            ["Conservative", "Steady", "Aggressive"],
            p=config.PERSONALITY_PROBS,
        )
        
        # Continuous emotion: 1.0 = ecstatic, 0.0 = completely frustrated
        self.emotion_val = np.clip(rng_agent.normal(config.EMOTION_INITIAL_MU, config.EMOTION_INITIAL_SIGMA), 0.1, 1.0)
        
        # Base patience drawn from lognormal distribution based on personality
        mu, sig = config.PATIENCE_PARAMS[self.personality]
        self.base_patience = rng_agent.lognormal(mu, sig)
        
        self.balk_threshold = config.BALK_THRESHOLDS[self.personality]

    @property
    def patience_threshold(self) -> float:
        """Effective patience is a function of base patience scaled by current emotion."""
        return max(2.0, self.base_patience * self.emotion_val)

    def decide_balk(self, visible_queue_length: int) -> bool:
        """Balking likelihood increases as initial emotion drops."""
        threshold = self.balk_threshold
        if self.emotion_val < 0.5:
            threshold = max(1, int(threshold * 0.6))
        return visible_queue_length >= threshold

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
