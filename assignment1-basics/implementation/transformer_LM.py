import torch
from torch import Tensor
import torch.nn as nn
from implementation.embedding import Embedding
from implementation.transformer_block import TransformerBlock
from implementation.rmsnorm import RMSNorm

class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        weights: dict[str, Tensor] | None = None,
        # in_indices: Tensor  # " batch_size sequence_length"
    ):
        super().__init__()
        self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=rope_theta,
                weights=None,
            ) for i in range(num_layers)
        ])

        # Final RMSNorm and LM head.
        self.ln_final = RMSNorm(d_model=d_model)
        self.lm_head = nn.Parameter(torch.empty((vocab_size, d_model)))
        torch.nn.init.normal_(self.lm_head, mean=0.0, std=0.02)
        if weights is not None:
            self.load_reference_weights(weights)

    def load_reference_weights(self, weights: dict[str, Tensor]) -> None:
        self.embedding.load_weight(weights["token_embeddings.weight"])
        for i, block in enumerate(self.transformer_blocks):
            block.load_reference_weights(
                {
                    "attn.q_proj.weight": weights[f"layers.{i}.attn.q_proj.weight"],
                    "attn.k_proj.weight": weights[f"layers.{i}.attn.k_proj.weight"],
                    "attn.v_proj.weight": weights[f"layers.{i}.attn.v_proj.weight"],
                    "attn.output_proj.weight": weights[f"layers.{i}.attn.output_proj.weight"],
                    "ln1.weight": weights[f"layers.{i}.ln1.weight"],
                    "ffn.w1.weight": weights[f"layers.{i}.ffn.w1.weight"],
                    "ffn.w2.weight": weights[f"layers.{i}.ffn.w2.weight"],
                    "ffn.w3.weight": weights[f"layers.{i}.ffn.w3.weight"],
                    "ln2.weight": weights[f"layers.{i}.ln2.weight"],
                }
            )
        with torch.no_grad():
            self.ln_final.g.copy_(weights["ln_final.weight"])
            self.lm_head.copy_(weights["lm_head.weight"])
    
    # token_ids: (batch_size, seq_len)
    def forward(self, token_ids: Tensor) -> Tensor:
        x = self.embedding(token_ids) # (batch_size, seq_len, d_model)
        for block in self.transformer_blocks:
            x = block(x) # (batch_size, seq_len, d_model)
        x = self.ln_final(x)
        
        logits = x @ self.lm_head.T # (batch_size, seq_len, vocab_size)
        return logits
