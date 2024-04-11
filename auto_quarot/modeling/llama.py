from .base import QuarotForCausalLM


class LlamaQuarotForCausalLM(QuarotForCausalLM):
    layers_block_name = "model.layers"
    rope_function_name = "apply_rotary_pos_emb"
    embeddings = ["model.embed_tokens"]
    lm_head = "lm_head"
    pre_head_layernorm = "model.norm"
    mlp_bottleneck_size = "config.intermediate_size"
    layernorm_fuses = {
        "post_attention_layernorm": ["mlp.up_proj", "mlp.gate_proj"],
        "input_layernorm": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
    }
    attention_inputs = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]
    attention_output = "self_attn.o_proj"
    mlp_inputs = ["mlp.up_proj", "mlp.gate_proj"]
    mlp_output = "mlp.down_proj"
    o_proj = "self_attn.o_proj"
    v_proj = "self_attn.v_proj"


__all__ = ["LlamaQuarotForCausalLM"]
