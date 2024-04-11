from .base import QuarotForCausalLM
from .llama import LlamaQuarotForCausalLM

QUAROT_MODEL_MAP = {
    "llama": LlamaQuarotForCausalLM,
    "mistral": LlamaQuarotForCausalLM,
}


class AutoQuarotForForCausalLM:
    @staticmethod
    def from_transformers(model) -> QuarotForCausalLM:
        cls = QUAROT_MODEL_MAP[model.config.model_type]
        return cls(model)
