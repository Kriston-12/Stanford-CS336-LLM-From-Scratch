import torch

class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.embedding = torch.nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        )
        # Important: torch.empty is uninitialized memory. Without explicit
        # initialization, values can be arbitrarily large and destabilize
        # training (loss may appear flat or become NaN).
        torch.nn.init.normal_(self.embedding, mean=0.0, std=0.02)

    # token_ids: (batch_size, seq_len)
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding[token_ids]
    

def test_token_index():
    a = torch.arange(1, 10).reshape(3, 3)
    print(a)
    print(a[2, 2])
