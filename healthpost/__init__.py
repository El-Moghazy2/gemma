"""HealthPost: Agentic CHW decision support powered by MedGemma.

Provides a complete patient visit workflow for Community Health Workers:
intake, diagnosis, prescription, and dispensing. Uses MedGemma for medical
vision and reasoning, MedASR for voice transcription, and a ReAct agent
for autonomous clinical reasoning.
"""

from .agent import AgentResult, MedicalAgent
from .config import Config
from .core import HealthPost

__version__ = "0.2.0"
__all__ = ["HealthPost", "Config", "MedicalAgent", "AgentResult"]
