import numpy as np
import torch


class DataLoaderLite:
    def __init__(self, data, batch_size, block_size, start_step=0):
        self.data = data
        self.batch_size = batch_size
        self.block_size = block_size
        self.tokens_per_batch = batch_size * block_size
        # If resuming from checkpoint, jump directly to that exact token position:
        self.current_position = start_step * self.tokens_per_batch

    def next_batch(self, device):
        B, T = self.batch_size, self.block_size
        req_len = B * T + 1

        # Wrap around to start if we reach the end of the dataset
        if self.current_position + req_len > len(self.data):
            self.current_position = 0

        # 1 single slice from the memmap (no python loops, no torch.stack)
        buf = self.data[self.current_position : self.current_position + req_len]
        self.current_position += B * T

        # Convert to tensor and reshape directly to (B, T)
        x = torch.from_numpy(buf[:-1].astype(np.int64)).view(B, T)
        y = torch.from_numpy(buf[1:].astype(np.int64)).view(B, T)

        return x.to(device), y.to(device)
