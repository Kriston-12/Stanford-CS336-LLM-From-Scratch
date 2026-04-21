
from __future__ import annotations
import torch

from implementation.transformer_LM import Transformer


def sample_next_token(
    logits: torch.Tensor,  # (batch_size, vocab_size)
    temperature: float = 1.0,
    top_p_threshold: float = 1.0,
) -> torch.Tensor:

    probs = torch.softmax(logits / temperature, dim=-1)

    if top_p_threshold < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        remove_mask = cumulative_probs > top_p_threshold

        # e.g. remove_mask = [False, False, True], we need all 3 elements to exceed the threshold
        # remove_mask[..., 1:] = [False, True]
        # remove_mask[..., :-1] = [False, False]
        remove_mask[..., 1:] = remove_mask[..., :-1].clone() # shift right by one
        remove_mask[..., 0] = False

        sorted_probs = sorted_probs.masked_fill(remove_mask, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

        # torch.multinomial return indices instead of values.
        sampled_sorted_positions = torch.multinomial(sorted_probs, num_samples=1)
        return torch.gather(sorted_indices, dim=-1, index=sampled_sorted_positions)

    return torch.multinomial(probs, num_samples=1)


def decode(
    model: Transformer,
    prompt_token_ids: torch.Tensor,  # (batch_size, sequence_length)
    max_number_of_tokens: int,
    temperature: float = 1.0,
    top_p_threshold: float = 1.0,
    end_token_ids: list[int] | None = None,
) -> torch.Tensor:

    generated = prompt_token_ids
    context_length = context_length = model.transformer_blocks[0].attn.max_seq_len
    eos_ids = None if end_token_ids is None else torch.tensor(end_token_ids, device=generated.device)

    for _ in range(max_number_of_tokens):
        model_input = generated if context_length is None else generated[:, -context_length:]
        logits = model(model_input)[:, -1, :]  # (batch_size, vocab_size)
        next_token_ids = sample_next_token(
            logits=logits,
            temperature=temperature,
            top_p_threshold=top_p_threshold,
        )

        generated = torch.cat([generated, next_token_ids], dim=1)

        # 这里应该去掉next_token_ids = eos的batch元素，但是为了简单，我们等到all元素都是eos才break
        if eos_ids is not None and torch.isin(next_token_ids, eos_ids).all():
            break

    return generated
