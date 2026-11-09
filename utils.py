import torch
import torch.nn as nns


def windowed_batch(batch, sequence_length=3, padding_value=0):

        left_padding = torch.full((batch.size(0), sequence_length-1),
                                padding_value, dtype=batch.dtype, 
                                device=batch.device)
        batch_with_padding = torch.cat([left_padding, batch], dim=1)

        return batch_with_padding.unfold(1, sequence_length, 1)