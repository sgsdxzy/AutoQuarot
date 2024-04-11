from typing import Dict, List, Literal

import tqdm
from ..utils import nested_attr, fuse_ln_linear, get_orthogonal_matrix
from ..hadamard_utils import apply_exact_had_to_linear, matmul_hadU_cuda_had
import torch


class QuarotForCausalLM:
    layers_block_name: str
    rope_function_name: str
    embeddings: List[str]
    lm_head: str
    pre_head_layernorm: str
    mlp_bottleneck_size: str
    layernorm_fuses: Dict[str, List[str]]
    attention_inputs: List[str]
    attention_output: str
    mlp_inputs: List[str]
    mlp_output: str
    o_proj: str
    v_proj: str

    def __init__(self, model):
        self.model = model

    def _get_rope_function_name(self):
        return nested_attr(self.model, self.rope_function_name)

    def _get_layers(self):
        return nested_attr(self.model, self.layers_block_name)

    def _get_embeddings(self) -> list[torch.nn.Module]:
        return [nested_attr(self.model, embeddings) for embeddings in self.embeddings]

    def _get_lm_head(self):
        return nested_attr(self.model, self.lm_head)

    def _get_pre_head_layernorm(self):
        return nested_attr(self.model, self.pre_head_layernorm)

    def _get_mlp_bottleneck_size(self):
        return nested_attr(self.model, self.mlp_bottleneck_size)

    def _rotate_embeddings(self, Q: torch.Tensor, device):
        # Rotate the embeddings.
        for W in self._get_embeddings():
            dtype = W.weight.data.dtype
            W_ = W.weight.data.to(device=device, dtype=torch.float64)
            W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)

    def _rotate_attention_inputs(self, layer, Q: torch.Tensor, device):
        # Rotate the WQ, WK and WV matrices of the self-attention layer.
        for W in [nested_attr(layer, m) for m in self.attention_inputs]:
            dtype = W.weight.dtype
            W_ = W.weight.to(device=device, dtype=torch.float64)
            W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)

    def _rotate_attention_output(self, layer, Q: torch.Tensor, device):
        # Rotate output matrix of the self-attention layer.
        W = nested_attr(layer, self.attention_output)
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=device, dtype=torch.float64)
        W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
        if W.bias is not None:
            b = W.bias.data.to(device=device, dtype=torch.float64)
            W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)

    def _rotate_mlp_input(self, layer, Q: torch.Tensor, device):
        # Rotate the MLP input weights.
        for W in [nested_attr(layer, m) for m in self.mlp_inputs]:
            dtype = W.weight.dtype
            W_ = W.weight.data.to(device=device, dtype=torch.float64)
            W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)

    def _rotate_mlp_output(self, layer, Q: torch.Tensor, device):
        # Rotate the MLP output weights and bias.
        W = nested_attr(layer, self.mlp_output)
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=device, dtype=torch.float64)
        W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
        apply_exact_had_to_linear(
            W, had_dim=-1, output=False
        )  # apply exact (inverse) hadamard on the weights of mlp output
        if W.bias is not None:
            b = W.bias.data.to(device=device, dtype=torch.float64)
            W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)

    def _rotate_faster_down_proj(self, layer, hardK):
        W = nested_attr(layer, self.mlp_output)
        dtype = W.weight.data.dtype
        W.weight.data = matmul_hadU_cuda_had(W.weight.data.float().cuda(), hardK)
        W.weight.data = W.weight.data.to(device="cpu", dtype=dtype)

    def _rotate_head(self, Q: torch.Tensor, device) -> None:
        # Rotate the head.
        W = self._get_lm_head()
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)

    def _rotate_ov_proj(self, layer, head_num, head_dim):
        v_proj = nested_attr(layer, self.v_proj)
        o_proj = nested_attr(layer, self.o_proj)
        apply_exact_had_to_linear(v_proj, had_dim=head_dim, output=True)
        apply_exact_had_to_linear(o_proj, had_dim=-1, output=False)

    @torch.inference_mode()
    def fuse_layer_norms(self):
        # Embedding fusion
        for W in self._get_embeddings():
            W_ = W.weight.data.double()
            W.weight.data = (W_ - W_.mean(dim=-1, keepdim=True)).to(W.weight.data.dtype)

        layers = self._get_layers()
        # Fuse the linear operations in Layernorm into the adjacent linear blocks.
        for layer in layers:
            # fuse the input layernorms into the linear layers
            for norm, linears in self.layernorm_fuses.items():
                fuse_ln_linear(
                    nested_attr(layer, norm),
                    [nested_attr(layer, linear) for linear in linears],
                )

        fuse_ln_linear(self._get_pre_head_layernorm(), [self._get_lm_head()])

    @torch.inference_mode()
    def rotate_model(self, rotate_mode: Literal["random", "hadamard"], device):
        Q = get_orthogonal_matrix(self.model.config.hidden_size, rotate_mode)
        config = self.model.config
        num_heads = config.num_attention_heads
        model_dim = config.hidden_size
        head_dim = model_dim // num_heads

        self._rotate_embeddings(Q, device=device)
        self._rotate_head(Q, device=device)
        layers = self._get_layers()
        for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating")):
            self._rotate_attention_inputs(layers[idx], Q, device=device)
            self._rotate_attention_output(layers[idx], Q, device=device)
            self._rotate_mlp_input(layers[idx], Q, device=device)
            self._rotate_mlp_output(layers[idx], Q, device=device)
            self._rotate_ov_proj(layers[idx], num_heads, head_dim)

        for param in self.model.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()


__all__ = ["QuarotForCausalLM"]
