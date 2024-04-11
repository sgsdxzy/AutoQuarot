import math
from typing import Dict, List, Literal

import fast_hadamard_transform
import torch
import tqdm

from ..hadamard_utils import (apply_exact_had_to_linear, get_hadK,
                              matmul_hadU_cuda, matmul_hadU_cuda_had)
from ..utils import fuse_ln_linear, get_orthogonal_matrix, nested_attr


class QuarotForCausalLM(torch.nn.Module):
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
        super().__init__()
        self._model = model

    @property
    def model(self):
        return self._model

    @property
    def quarot_config(self):
        if getattr(self._model.config, "quarot_config", None) is None:
            self._model.config.quarot_config = {
                "fused": False,
                "rotated": False,
            }
        return self._model.config.quarot_config

    def _get_rope_function_name(self):
        return nested_attr(self._model, self.rope_function_name)

    def _get_layers(self):
        return nested_attr(self._model, self.layers_block_name)

    def _get_embeddings(self) -> list[torch.nn.Module]:
        return [nested_attr(self._model, embeddings) for embeddings in self.embeddings]

    def _get_lm_head(self):
        return nested_attr(self._model, self.lm_head)

    def _get_pre_head_layernorm(self):
        return nested_attr(self._model, self.pre_head_layernorm)

    def _get_mlp_bottleneck_size(self):
        return nested_attr(self._model, self.mlp_bottleneck_size)

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
        if self.quarot_config["fused"]:
            raise RuntimeError("layernorms are already fused")

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

        self.quarot_config["fused"] = True

    @torch.inference_mode()
    def rotate_model(self, rotate_mode: Literal["random", "hadamard"], device):
        if self.quarot_config["rotated"]:
            raise RuntimeError("model is already rotated")

        Q = get_orthogonal_matrix(
            self._model.config.hidden_size, rotate_mode, device=device
        )
        config = self._model.config
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

        for param in self._model.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()

        self.quarot_config["rotated"] = True

    def _add_pre_down_proj_hook(self, layer):
        if getattr(layer, "_hadamard_hook", False):
            return

        def wrap(func):
            def wrapper(x: torch.Tensor):
                nonlocal had_K
                had_K = had_K.to(x.device)
                x = matmul_hadU_cuda(x, had_K, K)
                return func(x)

            return wrapper

        had_K, K = get_hadK(self.model.config.intermediate_size)
        layer.forward = wrap(layer.forward)
        layer._hadamard_hook = True

    def _add_pre_o_proj_hook(self, layer):
        if getattr(layer, "_hadamard_hook", False):
            return

        def wrap(func):

            def wrapper(x: torch.Tensor):
                # todo: implement this in QAttention to avoid reshaping!
                init_shape = x.shape
                if K == 1:
                    x = fast_hadamard_transform.hadamard_transform(
                        x.reshape(-1, init_shape[-1] // had_dim, had_dim).transpose(
                            1, 2
                        ),
                        scale=1 / math.sqrt(init_shape[-1] // had_dim),
                    ).transpose(1, 2)
                else:
                    nonlocal had_K
                    had_K = had_K.to(x.device)
                    x = (
                        had_K.to(x.dtype)
                        @ x.reshape(-1, init_shape[-1] // had_dim, had_dim)
                    ) / math.sqrt(init_shape[-1] // had_dim)
                x = x.reshape(init_shape)

                return func(x)

            return wrapper

        had_K, K = get_hadK(self.model.config.num_attention_heads)
        had_dim = self.model.config.hidden_size // self.model.config.num_attention_heads
        layer.forward = wrap(layer.forward)
        layer._hadamard_hook = True

    def add_pre_output_hooks(self):
        layers = self._get_layers()
        for layer in layers:
            down_proj = nested_attr(layer, self.mlp_output)
            self._add_pre_down_proj_hook(down_proj)
            o_proj = nested_attr(layer, self.attention_output)
            self._add_pre_o_proj_hook(o_proj)


__all__ = ["QuarotForCausalLM"]
