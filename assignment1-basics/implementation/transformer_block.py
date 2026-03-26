import torch
import torch.nn as nn
from implementation.swiglu import SwiGLU
from implementation.rmsnorm import RMSNorm
from implementation.multihead_attention import MultiHeadAttention

"""
Args:
    d_model (int): The dimensionality of the Transformer block input.
    num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
        evenly divisible by `num_heads`.
    d_ff (int): Dimensionality of the feed-forward inner layer.
    max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
    theta (float): RoPE parameter.
    weights (dict[str, Tensor]):
        State dict of our reference implementation.
        The keys of this dictionary are:
        - `attn.q_proj.weight`
            The query projections for all `num_heads` attention heads.
            Shape is (d_model, d_model).
            The rows are ordered by matrices of shape (num_heads, d_k),
            so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
        - `attn.k_proj.weight`
            The key projections for all `num_heads` attention heads.
            Shape is (d_model, d_model).
            The rows are ordered by matrices of shape (num_heads, d_k),
            so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
        - `attn.v_proj.weight`
            The value projections for all `num_heads` attention heads.
            Shape is (d_model, d_model).
            The rows are ordered by matrices of shape (num_heads, d_v),
            so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
        - `attn.output_proj.weight`
            Weight of the multi-head self-attention output projection
            Shape is (d_model, d_model).
        - `ln1.weight`
            Weights of affine transform for the first RMSNorm
            applied in the transformer block.
            Shape is (d_model,).
        - `ffn.w1.weight`
            Weight of the first linear transformation in the FFN.
            Shape is (d_model, d_ff).
        - `ffn.w2.weight`
            Weight of the second linear transformation in the FFN.
            Shape is (d_ff, d_model).
        - `ffn.w3.weight`
            Weight of the third linear transformation in the FFN.
            Shape is (d_model, d_ff).
        - `ln2.weight`
            Weights of affine transform for the second RMSNorm
            applied in the transformer block.
            Shape is (d_model,).
    in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """
class TransformerBlock(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        d_ff: int, 
        max_seq_len: int, 
        theta: float, 
        weights: dict[str, torch.Tensor]):

        super().__init__()
        self.attention_rmsnorm = RMSNorm(d_model=d_model)
        self.ffn_rmsnorm = RMSNorm(d_model=d_model)
        # self.rope = RoPE(d_k=num_heads, max_seq_len=max_seq_len)

        self.attention_block = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            theta=theta,
            max_seq_len=max_seq_len,
            q_proj_weight=weights["attn.q_proj.weight"],
            k_proj_weight=weights["attn.k_proj.weight"],
            v_proj_weight=weights["attn.v_proj.weight"],
            o_proj_weight=weights["attn.output_proj.weight"]
        )

        self.swiglu = SwiGLU(d_model=d_model, d_ff=d_ff)

        # load refernece weights into parameters
        # 没有下面这个no_grad会报错RuntimeError: a leaf Variable that requires grad is being used in an in-place operation.
        # 这是因为torch不允许requires_grad=True的参数被in-place修改，因为这相当于修改了计算图chain中的activation，那么后续的auto diff全都会因为这个数值被影响
        with torch.no_grad():
            self.attention_rmsnorm.g.copy_(weights["ln1.weight"])
            self.ffn_rmsnorm.g.copy_(weights["ln2.weight"])
            self.swiglu.w1.weight.copy_(weights["ffn.w1.weight"])
            self.swiglu.w2.weight.copy_(weights["ffn.w2.weight"])
            self.swiglu.w3.weight.copy_(weights["ffn.w3.weight"])
        
    def forward(self, in_features: torch.Tensor) -> torch.Tensor:
        """
        in_features: (batch, sequence_length, d_model)
        returns: (batch, sequence_length, d_model)
        """
        x = in_features

        x_norm = self.attention_rmsnorm(x)
        attn_out = self.attention_block(
            x_norm,
        )
        x = x + attn_out

        x_norm = self.ffn_rmsnorm(x)
        ffn_out = self.swiglu(x_norm)
        x = x + ffn_out

        return x