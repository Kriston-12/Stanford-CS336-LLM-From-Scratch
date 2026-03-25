import torch
import torch.nn as nn
from einops import einsum, rearrange
from implementation.scaled_dot_product_attention import ScaledDotProductAttention
from implementation.rope import RoPE

class MultiHeadAttention(nn.Module):
    def __init__(
            self, 
            d_model: int, 
            num_heads: int, 
            theta: float = None, 
            max_seq_len: int = None, 
            q_proj_weight: torch.Tensor = None,
            k_proj_weight: torch.Tensor = None,
            v_proj_weight: torch.Tensor = None,
            o_proj_weight: torch.Tensor = None,
            device=None,
            dtype=None
        ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scaled_dot_product_attention = ScaledDotProductAttention()
        if theta is not None and max_seq_len is not None:
            self.rope = RoPE(d_k=self.head_dim, theta=theta, max_seq_len=max_seq_len)
        else:
            self.rope = None

        self.q_proj = q_proj_weight
        self.k_proj = k_proj_weight
        self.v_proj = v_proj_weight
        self.o_proj = o_proj_weight

        # 注意causal 只跟seq_len有关而不是d_model
        # tensor.tril()表示下三角矩阵, tensor.triu()表示上三角矩阵
        causal = torch.ones(max_seq_len, max_seq_len, dtype=torch.bool).tril() # # True on and below diagonal
        self.register_buffer("causal_mask", causal, persistent=False)

    def _get_causal_mask(self, q_len: int, k_len: int, device) -> torch.Tensor:
        # (q_len, k_len), True=keep
        return self.causal_mask[:q_len, :k_len].to(device=device)

    #  " ... sequence_length d_in"
    def forward(self, in_features: torch.Tensor) -> torch.Tensor: 
        q = in_features @ self.q_proj.T # (..., seq_len, d_k)
        q = rearrange(q, "... seq_len (num_heads head_dim_q) -> ... num_heads seq_len head_dim_q", num_heads=self.num_heads)
        if self.rope is not None:
            q = self.rope(q)
        k = in_features @ self.k_proj.T # (..., seq_len, d_k)
        k = rearrange(k, "... seq_len (num_heads head_dim_k) -> ... num_heads seq_len head_dim_k", num_heads=self.num_heads)
        if self.rope is not None:
            k = self.rope(k)
        v = in_features @ self.v_proj.T # (..., seq_len, d_v)
        v = rearrange(v, "... seq_len (num_heads head_dim_v) -> ... num_heads seq_len head_dim_v", num_heads=self.num_heads)
        mask = self._get_causal_mask(q.shape[-2], k.shape[-2], q.device)
        attention_output = self.scaled_dot_product_attention(q, k, v, mask) # (..., num_heads, seq_len, head_dim_v)
        attention_output = rearrange(attention_output, "... num_heads seq_len head_dim_v -> ... seq_len (num_heads head_dim_v)") # (..., seq_len, d_v * num_heads)
        return einsum(attention_output, self.o_proj, "... seq_len d_v, d_model d_v -> ... seq_len d_model")