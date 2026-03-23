"""

https://github.com/philipphager/ultr-cm-vs-ips/blob/main/src/loss.py#L197
"""

import torch


def batch_pairs(x: torch.Tensor) -> torch.Tensor:
    """
    Creates i x j document pairs from batch or results.
    Adopted from pytorchltr

    Example:
        x = [
            [1, 2],
            [3, 4],
        ]

        [
            [[[1, 1], [1, 2]], [[2, 1], [2, 2]]],
            [[[3, 3], [3, 4]], [[4, 3], [4, 4]]]
        ]

    Args:
        x: Tensor of size (n_batch, n_results)

    Returns:
        Tensor of size (n_batch, n_results, n_results, 2) with all combinations
        of n_results.
    """

    if x.dim() == 2:
        x = x.reshape((x.shape[0], x.shape[1], 1))

    x_ij = torch.repeat_interleave(x, x.shape[1], dim=2)
    x_ji = torch.repeat_interleave(x.permute(0, 2, 1), x.shape[1], dim=1)

    return torch.stack([x_ij, x_ji], dim=3)


def mask_padding(x: torch.Tensor, n: torch.Tensor, fill: float = 0.0):
    n_batch, n_results = x.shape
    n = n.unsqueeze(-1)
    mask = torch.arange(n_results).repeat(n_batch, 1).type_as(x)
    x = x.float()
    x[mask >= n] = fill

    return x
