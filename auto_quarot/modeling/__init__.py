from .llama import LlamaQuarotForCausalLM
from .base import QuarotForCausalLM
from transformers import AutoModelForCausalLM

QUAROT_MODEL_MAP = {
    "llama": LlamaQuarotForCausalLM,
    "mistral": LlamaQuarotForCausalLM,
}

class AutoQuarotForForCausalLM:
    @staticmethod
    def from_transformers(model: AutoModelForCausalLM) -> QuarotForCausalLM:
        cls = QUAROT_MODEL_MAP[model.config.model_type]
        return cls(model)