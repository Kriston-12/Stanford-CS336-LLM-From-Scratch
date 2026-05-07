"""
Given a dataset (a 1D numpy array of integers) and a desired batch size and
context length, sample language modeling input sequences and their corresponding
labels from the dataset.

Args:
    dataset (np.array): 1D numpy array of integer token IDs in the dataset.
    batch_size (int): Desired batch size to sample.
    context_length (int): Desired context length of each sampled example.
    device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
        to place the sampled input sequences and labels on.

Returns:
    Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
    is the sampled input sequences, and the second tuple item is the corresponding
    language modeling labels.
"""
from numpy import ndarray
import torch
class DataLoader:
    def __init__(
        self, 
        dataset: ndarray, 
        batch_size: int, 
        context_length: int, 
        device: str
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.context_length = context_length
        self.device = device

    def get_batch(self):
        # randint high是exclusive的，所以本来就不会取到len(self.dataset) - self.context_length这个值，这样就保证了后面i + self.context_length + 1不会越界。
        start_indices = torch.randint(low=0, high=len(self.dataset) - self.context_length, size=(self.batch_size,))
        input_sequences = torch.stack([torch.from_numpy(self.dataset[i: i + self.context_length]) for i in start_indices], dim=0)
        label_sequences = torch.stack([torch.from_numpy(self.dataset[i + 1: i + self.context_length + 1]) for i in start_indices], dim=0)
        return input_sequences.pin_memory().to(self.device), label_sequences.pin_memory().to(self.device)

    def __len__(self):
        return len(self.dataset) // self.batch_size

def load_data(
    dataset: ndarray,
    batch_size: int,
    context_length:int,
    device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    start_indices = torch.randint(low=0, high=len(dataset) - context_length, size=(batch_size,))
    input_sequences = torch.stack([torch.from_numpy(dataset[i: i + context_length]) for i in start_indices], dim=0)
    label_sequences = torch.stack([torch.from_numpy(dataset[i + 1: i + 1 + context_length]) for i in start_indices], dim=0)
    return input_sequences.pin_memory().to(device), label_sequences.pin_memory().to(device)
