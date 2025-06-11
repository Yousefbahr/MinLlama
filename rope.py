from typing import Tuple
import torch

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Helper function to reshape frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)

def apply_rotary_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query and key tensors. The rotation to each token
    embedding is a function of that token's position in the sequence, head_dim, and theta.
    The input tensors are reshaped as complex numbers to simplify your implementation.

    Args:
        query (torch.Tensor): Query tensor to apply rotary embeddings.
                              Shape: (batch_size, seqlen, n_local_heads, self.head_dim)
        key (torch.Tensor): Key tensor to apply rotary embeddings.
                              Shape: (batch_size, seqlen, n_local_kv_heads, self.head_dim)
        head_dim (int): Dimension of each attention head.
        max_seq_len (int): Maximum sequence length supported by model.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """

    _, seqlen, _, _ = query.shape
    device = query.device
    # todo
    #
    # Please refer to slide 22 in https://phontron.com/class/anlp2024/assets/slides/anlp-05-transformers.pdf
    # and Section 3 in https://arxiv.org/abs/2104.09864.

    # reshape xq and xk to match the complex representation
    query_real, query_imag = query.float().reshape(query.shape[:-1] + (-1, 2)).unbind(-1)
    key_real, key_imag = key.float().reshape(key.shape[:-1] + (-1, 2)).unbind(-1)
    # This separates each query/key vector into its odd and even indices (assuming *one-indexing*).
    # query_real contains q_1, q_3, q_5, ... and query_imag contains q_2, q_4, q_6, ...

    # First, compute the trigonometric values in the second and fourth columns in
    # slide 22 (linked above).

    freqs = 1.0 / theta ** (2 * torch.arange(0, head_dim // 2, device=device).float() / head_dim)

    positions = torch.arange(max_seq_len, device=device).unsqueeze(1)

    angles = positions * freqs

    cos = torch.cos(angles)[:seqlen, :]
    sin = torch.sin(angles)[:seqlen, :]

    cos = cos.view(seqlen, 1, 1, head_dim // 2)
    sin = sin.view(seqlen, 1, 1, head_dim // 2)

    # Then, combine these trigonometric values with the tensors query_real, query_imag,
    # key_real, and key_imag.

    query_real = query_real.permute(1, 0, 2, 3)
    query_imag = query_imag.permute(1, 0, 2, 3)
    key_real = key_real.permute(1, 0, 2, 3)
    key_imag = key_imag.permute(1, 0, 2, 3)

    q_rotated_even = query_imag * cos - query_real * sin
    q_rotated_odd = query_imag * sin + query_real * cos

    k_rotated_even = key_imag * cos - key_real * sin
    k_rotated_odd = key_imag * sin + key_real * cos

    # raise NotImplementedError

    q_rotated_even = q_rotated_even.permute(1, 0, 2, 3)
    q_rotated_odd = q_rotated_odd.permute(1, 0, 2, 3)
    k_rotated_even = k_rotated_even.permute(1, 0, 2, 3)
    k_rotated_odd = k_rotated_odd.permute(1, 0, 2, 3)

    query_out = torch.stack((q_rotated_even, q_rotated_odd), dim=-1).flatten(-2)
    key_out = torch.stack((k_rotated_even, k_rotated_odd), dim=-1).flatten(-2)

    # Return the rotary position embeddings for the query and key tensors
    return query_out, key_out