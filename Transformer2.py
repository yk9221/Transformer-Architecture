import math
import random
from typing import Dict, Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset, random_split

SEED = 42


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


set_seed()

"""## Dataset

We will work on a simple sequential dataset that shift the elements of the input sequence.
"""

# don't change
data_args = {
    "num_samples": 1000,
    "seq_len": 32,
    "vocab_size": 20,
    "distance": 4,
}


class ShiftDataset(Dataset):
    """
    Shifts the elements of a sequence.
    """

    def __init__(self, num_samples=1000, seq_len=32, vocab_size=20, distance=4):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.distance = distance

        self.src = torch.randint(1, vocab_size, (num_samples, seq_len))
        self.tgt = torch.zeros_like(self.src)
        self.tgt[:, distance:] = self.src[:, :-distance]


    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {"inputs": self.src[idx], "targets": self.tgt[idx]}


def visualize_dataset(dataset, num_examples: int = 4):
    for i in range(num_examples):
        batch = dataset[i]

        print(f"\nExample {i + 1}:")
        print(f"Source: {' | '.join([f'{i:4d}' for i in batch['inputs'].tolist()])}")
        print(f"Target: {' | '.join([f'{i:4d}' for i in batch['targets'].tolist()])}")


if __name__ == "__main__":
    set_seed()
    data = ShiftDataset(**data_args)
    visualize_dataset(data, 4)

"""# Part 1 - Positional Embeddings (2.5 Points)

In class, we discussed various forms of position embeddings for self-attention in Transformers. In this part you will implement four positional embeddings, each worth 0.5 points:

1. Sinusodial encoding (Vaswani et al. 2017) --- The original Transformer's absolute position encoding. The original Transformer used fixed sinusoidal functions to encode absolute positions:
$$\text{PE}(pos, 2i) = \sin(pos / 10000^{2i/d_{model}})$$
$$\text{PE}(pos, 2i+1) = \cos(pos / 10000^{2i/d_{model}})$$
where pos is the position and i is the dimension. These encodings are added to the input embeddings before the first transformer layer.

2. Relative bias encoding in [T5](https://arxiv.org/abs/1910.10683) --- Learned relative position biases. T5 introduced **relative position biases** that are added directly to attention scores rather than to embeddings. The key innovation is **bucketing**: relative distances are grouped into buckets, with exact positions for small distances and logarithmic spacing for larger distances.
There is a single set of attention "bias" scalars ($e_\Delta$) shared across all layers. Each attention head has its own set of bias values. The offsets in the bias values are logarithmically spaced, and there are 32 total bias values.
- **Bucketing scheme**:
  - First half of buckets: exact positions (0, 1, 2, ..., 15)
  - Second half: logarithmically spaced (16-32, 32-64, 64-128, ...)
The bias is added to attention logits: $\text{softmax}(\frac{QK^T}{\sqrt{d}} + \text{bias})$

We provide more explanation on T5 buckets later in the homework.

3. [RoPE (Rotary Position Embedding)](https://arxiv.org/abs/2104.09864) (Su et al. 2021) --- Rotation-based relative encoding
RoPE applies position-dependent rotations to query and key vectors in 2D subspaces. For a position $m$, pairs of dimensions are rotated by angle $m\theta_i$, where $\theta_i = 10000^{-2i/d}$. The rotation can be expressed as:
$$\begin{bmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{bmatrix} \begin{bmatrix} q_{2i} \\ q_{2i+1} \end{bmatrix}$$

4. [ALIBI](https://arxiv.org/abs/2108.12409) (Press et al. 2021) --- Linear position relatigve position bias. Like the T5 relative position bias, ALiBi adds a simple linear bias based on distance: $-m \cdot (i - j)$, where $m$ is a learned head-specific scalar slope parameter and $(i-j)$ is the distance between positions.


"""

def sinusoidal_position_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    res = torch.zeros((seq_len, d_model))
    ########################### YOUR CODE ###################################
    ## TODO: Compute sinusoidal position encodings
    ## PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    ## PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    ## Hint: Returns tensor of shape (seq_len, d_model) containing position encodings

    for pos in range(seq_len):
      for i in range(d_model // 2):
        res[pos][2 * i] = np.sin(pos / 10000 ** (2 * i / d_model))
        res[pos][2 * i + 1] = np.cos(pos / 10000 ** (2 * i / d_model))
    #########################################################################
    return res

"""**T5 Buckets Explanation**

- 1: Split buckets for bidirectional distances
  - `bidirectional=True`, so half of the buckets are for negative relative positions and half for positive relative positions.  
  - `num_buckets = 8` $\to$ 4 buckets for negative, 4 buckets for positive.  

- 2: Map negative positions
  - Negative relative positions (`query < key`) are mapped to the lower half of the buckets: `[0, 1, 2, 3]`.  
  - Zero distance (`query == key`) is treated as the smallest positive distance (usually mapped to the first (0) positive bucket).  

- 3: Map positive positions
  - Positive relative positions (`query > key`) are mapped to the upper half of the buckets: `[4, 5, 6, 7]`.  

- 4: Exact vs logarithmic buckets
  - Small distances (i.e., distances < num_buckets//2) are assigned to exact buckets.  
  - Larger distances (beyond this range) are mapped logarithmically to the remaining buckets, which allows covering larger relative positions efficiently without needing a bucket for every possible distance.

  We provide you with a sanity check in the second cell and you can find more test cases on Markus.

"""

def t5_relative_position_bucket(
    relative_position: torch.Tensor,
    bidirectional: bool = True,
    num_buckets: int = 32,
    max_distance: int = 128,
) -> torch.Tensor:
    """
    Helper function
    Translate relative positions to bucket indices.

    Buckets are divided into two parts:
    1. Exact buckets: for small relative distances  (0, 1, 2, ..., num_buckets//2 - 1)
    2. Logarithmic buckets: for larger relative distances up to max_distance

    If bidirectional=True, positive and negative positions get separate sets of buckets.

    Args:
        relative_position: Tensor of relative positions (query_pos - key_pos)
        bidirectional: If True, use separate buckets for positive/negative distances
        num_buckets: Total number of buckets,
            You can assume num_buckets is always even and larger than 2
        max_distance: Maximum distance to consider
    """
    relative_buckets = torch.zeros((relative_position.shape), dtype=torch.int)
    ########################### YOUR CODE ###################################
    ## TODO: Compute T5 relative embeddings
    # 1: Handle bidirectional positions
    # - Split num_buckets evenly for negative/positive positions
    # - Negative positions map to lower half, positive to upper half
    if bidirectional:
        # One way to do this is adding offset to positive positions
        ## Hint: modify relative positions and
        # work with absolute distances for bucket assignment
        num_buckets //= 2
        relative_buckets += (relative_position > 0) * num_buckets
        relative_position = torch.abs(relative_position)
    else:
        # For unidirectional, only consider non-positive distances and then
        # modify them to work with absolute distances for bucket assignment
        relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))

    # 2: Assign small distances to exact buckets (half of the buckets)
    # 3: The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
    # Hint: Make sure to clip to maximum bucket index
    # Hint: Combine small and large distance assignments

    max_exact = num_buckets // 2
    is_small = relative_position < max_exact
    val_if_large = max_exact + (torch.log(relative_position.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)).long()
    val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
    relative_buckets += torch.where(is_small, relative_position, val_if_large)
    #########################################################################
    assert relative_buckets.shape == relative_position.shape, (
        f"Relative buckets should have shape {relative_position.shape} got {relative_buckets.shape}"
    )
    assert (relative_buckets < 0).sum() == 0, "Relative buckets should be >= 0"
    return relative_buckets

def t5_relative_position_bias_with_values(
    query_len: int,
    key_len: int,
    bias_values: torch.Tensor,  # (num_heads, num_buckets) - learned parameters
    num_buckets: int = 32,
    max_distance: int = 128,
    bidirectional: bool = True,
) -> torch.Tensor:
    """
    Compute T5-style relative position biases using PROVIDED bias values.
    """
    query_positions = torch.arange(query_len).unsqueeze(1) # (query_len, 1)
    ########################### YOUR CODE ###################################
    ## TODO: implement key_positions and relative_position,
    ## query_positions are already provided to you as a hint
    key_positions = torch.arange(key_len).unsqueeze(0)
    relative_position = query_positions - key_positions
    #########################################################################
    # don't change the rest
    buckets = t5_relative_position_bucket(
        relative_position,
        bidirectional=bidirectional,
        num_buckets=num_buckets,
        max_distance=max_distance,
    )
    # Use the provided learned bias values
    res = bias_values[:, buckets]  # (num_heads, query_len, key_len)
    return res

## sanity check
if __name__ == "__main__":
    seq_len_ = 5
    num_buckets_ = 8
    relative_position = torch.tensor([
        [ 0, -1, -2, -3, -4],
        [ 1,  0, -1, -2, -3],
        [ 2,  1,  0, -1, -2],
        [ 3,  2,  1,  0, -1],
        [ 4,  3,  2,  1,  0],
    ])


    buckets = t5_relative_position_bucket(relative_position, bidirectional=True,
                                          num_buckets=num_buckets_, max_distance=3)
    expected = torch.tensor([[0, 1, 2, 3, 3],
        [5, 0, 1, 2, 3],
        [6, 5, 0, 1, 2],
        [7, 6, 5, 0, 1],
        [7, 7, 6, 5, 0]], dtype=torch.int32)
    assert torch.allclose(buckets, expected)

def alibi_bias(query_len: int, key_len: int, num_heads: int = 8) -> torch.Tensor:
    """
    Compute ALiBi position biases.
    Returns:
        Tensor of shape (num_heads, query_len, key_len) containing linear biases
    """
    res = torch.zeros((num_heads, query_len, key_len))
    ########################### YOUR CODE ###################################
    # TODO: Compute ALiBi biases. The shapes below are hints, you can compute the same function in other ways
    # 1. Compute slopes for each head: 2^(-8*(h+1)/num_heads)
    h = torch.arange(num_heads)
    slopes = torch.pow(2, -8 * (h + 1) / num_heads).unsqueeze(1).unsqueeze(2)
    # 2. Create distance matrix: relative_positions_{ij} = (i - j) for all query pos. i and key pos. j
    # Hint: past positions (j < i) give negative distances
    #           future positions (j > i) give positive distances
    ## TODO:
    query_positions = torch.arange(query_len).unsqueeze(1)
    key_positions = torch.arange(key_len).unsqueeze(0)
    relative_positions = query_positions - key_positions
    #########################################################################
    # 3. Apply slopes: -slope * distance for each head
    # Hint: For past tokens (distance < 0): this gives positive bias (slope * distance)
    # For future tokens (distance > 0): this gives negative bias
    # don't change
    res = -slopes * relative_positions  # (num_heads, query_len, key_len)
    return res

# torch.manual_seed(0)
# x = torch.round(torch.randn(1, 2, 4), decimals=2)
# *batch, seq_len, dim = x.shape
# m = torch.arange(seq_len)
# res = x.view(*batch, seq_len, dim // 2, 2, 1)

# i = torch.arange(0, dim // 2)
# theta = 10000 ** (-2 * i / dim)
# m_theta = m.unsqueeze(-1).float() @ theta.unsqueeze(0)
# cos_m_theta = torch.cos(m_theta)
# sin_m_theta = torch.sin(m_theta)

# rotation_matrix = torch.stack([ torch.stack([cos_m_theta, -sin_m_theta], dim=-1), torch.stack([sin_m_theta,  cos_m_theta], dim=-1) ], dim=-2)
# print(rotation_matrix.shape)
# (rotation_matrix @ res).squeeze(-1).view(*batch, seq_len, dim)

def apply_rotary_position_embedding(
    x: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """
    Apply rotary position embedding to input tensor.
    Args:
        x: Input tensor of shape (..., seq_len, dim)
        positions: Position indices of shape (seq_len,)

    Returns:
        Tensor of same shape as x with rotary embeddings applied
    """
    res = x.clone()
    *batch_dims, seq_len, dim = x.shape
    ########################### YOUR CODE ###################################
    # TODO: Apply rotary position embeddings.
    # 1. Compute rotation frequencies: theta_i = 10000^(-2i/dim)
    # 2. Compute angles: m * theta_i for each position m
    # 3. (Hint) Reshape x into pairs: (..., seq_len, dim//2, 2)
    # 4. Apply rotation matrix to each pair
    # 5. Reshape back to original shape

    i = torch.arange(0, dim // 2)
    theta = 10000 ** (-2 * i / dim)
    m_theta = positions.unsqueeze(-1).float() @ theta.unsqueeze(0)
    cos_m_theta = torch.cos(m_theta)
    sin_m_theta = torch.sin(m_theta)

    res = res.view(*batch_dims, seq_len, dim // 2, 2, 1)
    rotation_matrix = torch.stack([ torch.stack([cos_m_theta, -sin_m_theta], dim=-1), torch.stack([sin_m_theta,  cos_m_theta], dim=-1) ], dim=-2)
    res = (rotation_matrix @ res).squeeze(-1).view(*batch_dims, seq_len, dim)

    #########################################################################
    return res

"""## Part 2 - Implement the transformer

- Pre vs. post layernorm (0.5 points)
- Norm type: layernorm and rmsnorm (0.5 points each)
- Correct transformer implementation (0.8 points)
"""

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (used in LLaMA)"""

    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        rms = x
        ########################### YOUR CODE ###################################
        ## TODO: RMS = sqrt(mean(x^2) + eps)
        ## Hint: RMSNorm operates on the last dimension
        rms = torch.sqrt(self.eps + torch.mean(x ** 2, dim=-1)).unsqueeze(-1)
        rms = x * self.weight / rms
        #########################################################################
        return rms


class LayerNorm(nn.Module):
    """
    Layer Normalization (Ba et al., 2016)

    Normalizes across the feature dimension:
    y = gamma * (x - mean) / sqrt(var + eps) + beta

    Used in original Transformer (Vaswani et al., 2017)
    """

    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))  # gamma
        self.bias = nn.Parameter(torch.zeros(d_model))  # beta

    def forward(self, x):
        res = x
        ########################### YOUR CODE ###################################
        # TODO: Implement Layer Normalization
        # 1. Compute mean, Hint: across feature dimension (dim=-1)
        # 2. Compute variance across feature dimension
        # 3. Normalize: (x - mean) / sqrt(var + eps)
        # 4. Apply affine transformation: weight * normalized + bias

        mean = torch.mean(x, dim=-1, keepdim=True)
        var = torch.var(x, dim=-1, keepdim=True, unbiased=False)
        res = (x - mean) / torch.sqrt(var + self.eps)
        res = self.weight * res + self.bias
        #########################################################################
        return res

# ============================================================================
# Tiny Transformer
# ============================================================================


class TinyTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=64,
        nhead=4,
        num_layers=2,
        max_len=128,
        pos_encoding="rope",
        num_t5_buckets: int = 32,
        norm_type: Literal["rmsnorm", "layernorm"] = "layernorm",
        pre_norm: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.pos_encoding_type = pos_encoding
        self.max_len = max_len

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [
                TransformerLayer(d_model, nhead, pos_encoding, norm_type, pre_norm)
                for _ in range(num_layers)
            ]
        )
        self.output = nn.Linear(d_model, vocab_size)

        ########################### YOUR CODE ###################################
        # TODO: Create t5_bias if pos_encoding is t5
        # Hint: T5 biases should be learnable parameters of shape (nhead, num_t5_buckets)
        self.t5_bias = None
        if pos_encoding == "t5":
            self.t5_bias = nn.Parameter((2 * torch.randn(nhead, num_t5_buckets) - 1) * np.sqrt(6 / (nhead + num_t5_buckets)))
        #########################################################################

    def forward(self, x):
        x = self.embedding(x)

        ########################### YOUR CODE ###################################
        # TODO: Add sinusoidal position encodings if pos_encoding_type is "sinusoidal"
        if self.pos_encoding_type == "sinusoidal":
            ## Hint: Make sure to cast the pos_enc to shape (1, seq_len, d_model)
            pos_encoding = sinusoidal_position_encoding(x.shape[1], self.d_model).unsqueeze(0)
            x = x + pos_encoding
        #########################################################################

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, t5_biases=self.t5_bias)

        return self.output(x)  # (batch, seq_len, vocab_size)

    @staticmethod
    def loss_fn(y_hat: torch.Tensor, y: torch.Tensor):
        res = 0.0
        ########################### YOUR CODE ###################################
        ## TODO: Implement cross-entropy loss
        ## Hint: y has class labels (e.g. 1, 2, 3, 4, ...), it is not one-hot encoded
        res = F.cross_entropy(y_hat.transpose(-2, -1), y)
        #########################################################################
        return res


class TransformerLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        pos_encoding="rope",
        norm_type: Literal["layernorm", "rmsnorm"] = "layernorm",
        pre_norm: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.pos_encoding = pos_encoding

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        ##########################################
        ## Initialize normalization layers based on norm_type
        if norm_type == "layernorm":
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
        elif norm_type == "rmsnorm":
            self.norm1 = RMSNorm(d_model)
            self.norm2 = RMSNorm(d_model)
        else:
            raise ValueError(f"Only norms available are RMSNorm and LayerNorm")
        ##########################################

        self.pre_norm = pre_norm
        ########################### YOUR CODE ###################################
        ## TODO: Create feed forward layers
        ## Linear -> GeLU -> Linear, FFN inner_dimension should be 4 d_model
        ## Hint: Make sure that your ffn is wrapped in nn.Sequential
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )
        #########################################################################

    def forward(self, x, t5_biases):
        ########################### YOUR CODE ###################################
        # TODO: Implement pre-norm or post-norm architecture
        # Pre-norm (modern): x = x + sublayer(norm(x))
        # Post-norm (original): x = norm(x + sublayer(x))
        # Hint: Check self.pre_norm to decide which to use
        ##########################################
        if self.pre_norm:
            ## TODO: Pre-norm: normalize before sublayer
            x = x + self.self_attention(self.norm1(x), t5_biases)
            x = x + self.ffn(self.norm2(x))
        else:
            ## TODO: Post-norm: normalize after residual
            x = self.norm1(x + self.self_attention(x, t5_biases))
            x = self.norm2(x + self.ffn(x))
        #########################################################################
        return x

    def self_attention(self, x, t5_biases):
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)

        if self.pos_encoding == "rope":
          positions = torch.arange(seq_len, device=x.device)
          ########################### YOUR CODE ###################################
          ## TODO: Apply RoPE if pos_encoding is "rope"
          q = apply_rotary_position_embedding(q, positions)
          k = apply_rotary_position_embedding(k, positions)
          #########################################################################

        # Compute attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)

        ########################### YOUR CODE ###################################
        ## TODO: Add position biases based on pos_encoding type
        ## ALiBi: Use alibi_bias(seq_len, seq_len, self.nhead)
        ## T5: Use t5_relative_position_bias_with_values(seq_len, seq_len, t5_biases)
        ## Hint: Add the bias to scores before softmax
        ## Hint: Move bias to same device as scores
        ##########################################
        if self.pos_encoding == "alibi":
            bias = alibi_bias(seq_len, seq_len, self.nhead)
            scores += bias
        elif self.pos_encoding == "t5":
            bias = t5_relative_position_bias_with_values(seq_len, seq_len, t5_biases)
            scores += bias
        #########################################################################

        # Apply causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0) == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        # Reshape back
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(out)

"""## Trainer (0.2 points)
Implement the trainer, this will be quite similar to Hw 6 and 7 trainer with small tweaks:
- We won't use attention_mask
- We will keep track of accuracy
"""

class Trainer:
    def __init__(
        self,
        max_epochs,
        batch_size,
        gradient_clip_val=1,
        device="cpu",
        print_every: int = 5,
    ):
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.gradient_clip_val = gradient_clip_val
        self.device = device
        self.train_loss = []  # record the avg. batch loss every epoch
        self.valid_loss = []  # record the avg. batch loss every epoch
        self.train_acc = []  # record the avg. batch accuracy every epoch
        self.val_acc = []  # record the avg. batch accuracy every epoch
        self.print_every = print_every

    @staticmethod
    def clip_gradients(model, max_norm):
        if not max_norm:
            return
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

    def get_dataloader(self, data):
        g = torch.Generator()
        g.manual_seed(SEED)
        train_size = int(0.8 * len(data))
        train_data, val_data = random_split(data, [train_size, len(data) - train_size], generator=g)
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, generator=g)
        valid_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False, generator=g)

        return train_loader, valid_loader

    def fit(self, model, data, optimizer=None):
        model.to(self.device)
        if optimizer is None:
            optimizer = torch.optim.SGD(model.parameters(), lr=model.lr)
        train_loader, valid_loader = self.get_dataloader(data)

        for epoch in range(self.max_epochs):
            model.train()
            train_loss = 0
            valid_loss = 0
            correct = 0
            total = 0
            ########################### YOUR CODE ###################################
            # TODO: Train the model for max_epochs many steps
            # Complete a single forward and backward pass on a given training batch
            # Record the training loss and training accuracy
            for batch in train_loader:
                # Extract from dictionary
                X = batch["inputs"].to(self.device)
                Y = batch["targets"].to(self.device)
                ## TODO: Training logic
                ## TODO: compute accuracy

                optimizer.zero_grad()

                y_hat = model(X)
                loss = model.loss_fn(y_hat, Y)
                loss.backward()
                self.clip_gradients(model, self.gradient_clip_val)
                optimizer.step()


                train_loss += loss.item()

                predictions = torch.argmax(y_hat, dim=-1)
                correct += (predictions == Y).sum().item()
                total += Y.numel()
            ########################################################################
            assert 0.0 <= correct / total <= 1.0, "Accuracy should be between 0 and 1"
            self.train_loss.append(train_loss / len(train_loader))
            self.train_acc.append(correct / total)

            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                ########################### YOUR CODE ###################################
                # TODO: at the end of each epoch, evaluate the model on the validation set.
                # Complete a single forward pass on a given validation batch
                # Record the validation loss and accuracy
                for batch in valid_loader:
                    X = batch["inputs"].to(self.device)
                    Y = batch["targets"].to(self.device)

                    y_hat = model(X)
                    loss = model.loss_fn(y_hat, Y)
                    valid_loss += loss.item()

                    predictions = torch.argmax(y_hat, dim=-1)
                    val_correct += (predictions == Y).sum().item()
                    val_total += Y.numel()

                ########################################################################
            self.valid_loss.append(valid_loss / len(valid_loader))
            self.val_acc.append(val_correct / val_total)

            if (epoch + 1) % self.print_every == 0:
                print(
                    f"Epoch {epoch + 1} train loss: {self.train_loss[-1]:.5f},\t train acc: {self.train_acc[-1]:.5f}\n\tvalidation loss {self.valid_loss[-1]:.5f}, val acc: {self.val_acc[-1]:.5f} "
                )

    def predict(self, model, dataloader):
        """
        Generate predictions on a dataset.
        Returns:
            predictions, targets, avg_loss
        """
        model.to(self.device)
        model.eval()

        all_predictions = []
        all_targets = []
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            ########################### YOUR CODE ###################################
            # TODO: Generate predictions for all batches in the dataloader
            # 1. Extract inputs and targets from batch
            # 2. Move tensors to device
            # 3. Get model predictions
            # 4. Store predictions and targets, make sure to store predictions as the index of the predicted class
            # 5. Compute loss
            # 6. Compute accuracy

            for batch in dataloader:
                # Extract batch data
                X = batch["inputs"].to(self.device)
                Y = batch["targets"].to(self.device)

                y_hat = model(X)
                loss = model.loss_fn(y_hat, Y)

                total_loss += loss.item()

                predictions = torch.argmax(y_hat, dim=-1)
                all_predictions.append(predictions)
                all_targets.append(Y)

                num_batches += 1


            ########################################################################

        # Concatenate all predictions and targets
        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)
        avg_acc = (predictions == targets).float().mean().item()

        avg_loss = total_loss / num_batches
        return predictions, targets, avg_loss, avg_acc

# don't change
def visualize_training(
    trainer, data, positional_encoding_type, model=None, vocab_size=20, title=""
):
    tr, val = trainer.get_dataloader(data)
    preds, targets, avg_loss, avg_acc = trainer.predict(model, val)

    fig, axs = plt.subplots(1, 4, figsize=(20, 4))
    fig.suptitle(
        f"{title} - Val Acc: {avg_acc:.4f}",
        fontsize=14,
        fontweight="bold",
    )

    # 1. Confusion matrix
    cm = confusion_matrix(
        targets.flatten(), preds.flatten(), labels=np.arange(1, vocab_size)
    )
    im = axs[0].imshow(cm, cmap="Blues", aspect="auto")
    axs[0].set_xlabel("Predicted Token", fontsize=11)
    axs[0].set_ylabel("Target Token", fontsize=11)
    axs[0].set_title("Confusion Matrix", fontsize=12)
    plt.colorbar(im, ax=axs[0], fraction=0.046)

    # 2. Training and validation loss
    axs[1].plot(trainer.train_loss, label="Train Loss", linewidth=2)
    axs[1].plot(trainer.valid_loss, label="Val Loss", linewidth=2)
    axs[1].set_ylabel("Loss", fontsize=11)
    axs[1].set_xlabel("Epoch", fontsize=11)
    axs[1].set_title("Training & Validation Loss", fontsize=12)
    axs[1].legend()
    axs[1].grid(alpha=0.3)

    # 3. Training and validation accuracy
    axs[2].plot(trainer.train_acc, label="Train Acc", linewidth=2)
    axs[2].plot(trainer.val_acc, label="Val Acc", linewidth=2)
    axs[2].set_ylabel("Accuracy", fontsize=11)
    axs[2].set_xlabel("Epoch", fontsize=11)
    axs[2].set_title("Training & Validation Accuracy", fontsize=12)
    axs[2].set_ylim([0, 1.05])
    axs[2].legend()
    axs[2].grid(alpha=0.3)

    # 4. Positional encoding visualization (if applicable)
    if positional_encoding_type == "t5" and model is not None:
        # Visualize learned T5 biases
        seq_len = 64
        t5_bias = t5_relative_position_bias_with_values(
            seq_len, seq_len, model.t5_bias.detach().cpu()
        )
        im = axs[3].imshow(
            t5_bias[0].numpy(), cmap="RdBu", aspect="auto", vmin=-3, vmax=3
        )
        axs[3].set_xlabel("Key Position", fontsize=11)
        axs[3].set_ylabel("Query Position", fontsize=11)
        axs[3].set_title("Learned T5 Bias (Head 0)", fontsize=12)
        axs[3].plot([0, seq_len - 1], [0, seq_len - 1], "k--", linewidth=1, alpha=0.3)
        plt.colorbar(im, ax=axs[3], fraction=0.046)

    elif positional_encoding_type == "alibi":
        # Visualize ALiBi biases
        seq_len = 64
        alibi = alibi_bias(seq_len, seq_len, model.layers[0].nhead if model else 8)
        im = axs[3].imshow(alibi[0].numpy(), cmap="RdBu", aspect="auto")
        axs[3].set_xlabel("Key Position", fontsize=11)
        axs[3].set_ylabel("Query Position", fontsize=11)
        axs[3].set_title("ALiBi Bias (Head 0)", fontsize=12)
        axs[3].plot([0, seq_len - 1], [0, seq_len - 1], "k--", linewidth=1, alpha=0.3)
        plt.colorbar(im, ax=axs[3], fraction=0.046)

    elif positional_encoding_type == "rope" and model is not None:
        # Visualize RoPE attention pattern
        seq_len = 64
        d_model = model.d_model
        q = torch.randn(1, seq_len, d_model)
        k = torch.randn(1, seq_len, d_model)
        positions = torch.arange(seq_len)
        q_rope = apply_rotary_position_embedding(q, positions)
        k_rope = apply_rotary_position_embedding(k, positions)
        scores = torch.matmul(q_rope, k_rope.transpose(-2, -1)) / np.sqrt(d_model)
        im = axs[3].imshow(scores[0].numpy(), cmap="RdBu", aspect="auto")
        axs[3].set_xlabel("Key Position", fontsize=11)
        axs[3].set_ylabel("Query Position", fontsize=11)
        axs[3].set_title("RoPE Attention Pattern", fontsize=12)
        axs[3].plot([0, seq_len - 1], [0, seq_len - 1], "k--", linewidth=1, alpha=0.3)
        plt.colorbar(im, ax=axs[3], fraction=0.046)

    elif positional_encoding_type == "sinusoidal":
        # Visualize sinusoidal encoding
        seq_len = 64
        sin_enc = sinusoidal_position_encoding(seq_len, model.d_model if model else 64)
        im = axs[3].imshow(sin_enc.T.numpy(), cmap="RdBu", aspect="auto")
        axs[3].set_xlabel("Position", fontsize=11)
        axs[3].set_ylabel("Embedding Dimension", fontsize=11)
        axs[3].set_title("Sinusoidal Encoding", fontsize=12)
        plt.colorbar(im, ax=axs[3], fraction=0.046)

    fig.tight_layout()
    return fig

"""### Train the models and examine the output
Unlike previous homeworks, we won't ask you to optimize the hyper-parameters. If your implementation is correct, with the given hyper-parameters, you should achieve >0.8 accuracy with sinussoidal, T5, and RoPE. ~0.3 with no encodings and ALiBi.

(ungraded)

After running the training code and examining the visualizations, answer the following question:

1. Can the model learn the shift task without any position encoding (pos_encoding="none")?
   Why or why not? What does this tell you about the importance of position information?
1. Why does the ALiBi model perform poorly? What tweaks could have improved the performance?
1. Are certain positional embeddings better suited for certain datasets? (You can also experiment with three other datasets included in the next cells).
1. Examine the encoding heatmaps (4th column). Describe the patterns, are they similar to how you would expect them to be?

**Task-specific**
1. The shift task requires outputting token 0 for the first `distance` positions.
   Can you identify if the model learned this from the confusion matrix?
1. The shift task is a "relative position" task (copy from distance=4 back).
   Explain why relative position encodings (RoPE, T5, ALiBi) should theoretically
   perform better than absolute encodings (Sinusoidal) on this task.
1. If you trained on sequences of length 16, which encoding would you expect to
   perform best on sequences of length 32? Why?

"""

# don't change,
args = {
    "lr": 3e-4,
    "d_model": 64,
    "nhead": 8,
    "batch_size": 32,
    "num_epochs": 25,
    "gradient_clip_val": 1.0,
}

if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Configuration
    set_seed()
    data = ShiftDataset(
        data_args["num_samples"],
        data_args["seq_len"],
        data_args["vocab_size"],
        data_args["distance"],
    )
    for positional_encoding_type in ["none", "sinusoidal", "rope", "t5", "alibi"]:
        print("=" * 30 + positional_encoding_type + "=" * 30)
        set_seed()
        model = TinyTransformer(
            data.vocab_size,
            d_model=args["d_model"],
            nhead=args["nhead"],
            num_layers=2,
            max_len=data_args["seq_len"],
            pos_encoding=positional_encoding_type,
            norm_type="rmsnorm",
            pre_norm=True,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=args["lr"])

        trainer = Trainer(
            batch_size=args["batch_size"],
            max_epochs=args["num_epochs"],
            gradient_clip_val=args["gradient_clip_val"],
            device=device,
        )

        trainer.fit(model, data, optimizer)

        visualize_training(
            trainer, data, positional_encoding_type, model, data.vocab_size
        )
plt.show()

"""(Ungraded)

If you would like, you can experiment with different datasets below.
"""

# don't change
class ReversalDataset(Dataset):
    """
    Reverse the input sequence.

    Input:  [a, b, c, d, e, f, g, h]
    Output: [h, g, f, e, d, c, b, a]

    Requires knowing absolute positions to map position i → position (n-1-i)
    """

    def __init__(self, num_samples=1000, seq_len=16, vocab_size=20):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        src = torch.randint(1, self.vocab_size, (self.seq_len,))
        tgt = torch.flip(src, dims=[0])  # Reverse the sequence

        return {"inputs": src, "targets": tgt}


class PositionDependentDataset(Dataset):
    """
    Each position in the sequence should output a specific token based on position modulo vocab_size.

    Example (vocab_size=10):
    Position:  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ...]
    Target:    [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2,  3,  ...]

    Input is random, output depends only on absolute position!
    """

    def __init__(self, num_samples=1000, seq_len=32, vocab_size=10):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Input is random noise
        src = torch.randint(1, self.vocab_size, (self.seq_len,))

        # Target depends ONLY on absolute position
        positions = torch.arange(self.seq_len)
        tgt = (positions % (self.vocab_size - 1)) + 1  # Cycle through 1 to vocab_size-1

        return {"inputs": src, "targets": tgt}

"""# Part 2 - Architectural Changes - Train

For this part of the assignment we will use a different dataset, where we sort the numbers. There is actually nothing to change here (you have completed this part when implementing the architecture and norm layers) but make sure that your implementation is correct.
"""

class SortingDataset(Dataset):
    """
    Sort sequences
    """

    def __init__(self, num_samples=1000, seq_len=32, vocab_size=20):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        src = torch.randint(1, self.vocab_size, (self.seq_len,))
        tgt = torch.sort(src)[0]

        return {"inputs": src, "targets": tgt}

# don't change
if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Configuration
    set_seed()
    data = SortingDataset(
        data_args["num_samples"], data_args["seq_len"], data_args["vocab_size"]
    )
    for norm_type in ["rmsnorm", "layernorm"]:
        for pre_norm in [True, False]:
            positional_encoding_type = "rope"
            # for positional_encoding_type in ["sinusoidal", "rope", "t5", "alibi"]:
            title = "Pre-LN" if pre_norm else "Post-LN"
            title = f"{norm_type} - {title} - {positional_encoding_type}"
            print("=" * 30 + title + "=" * 30)
            set_seed()
            model = TinyTransformer(
                data.vocab_size,
                d_model=args["d_model"],
                nhead=args["nhead"],
                num_layers=2,
                max_len=data_args["seq_len"],
                pos_encoding=positional_encoding_type,
                norm_type=norm_type,
                pre_norm=pre_norm,
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=args["lr"])

            trainer = Trainer(
                batch_size=args["batch_size"],
                max_epochs=args["num_epochs"],
                gradient_clip_val=args["gradient_clip_val"],
                device=device,
            )

            trainer.fit(model, data, optimizer)

            visualize_training(trainer, data, positional_encoding_type, model, data.vocab_size, title)
plt.show()

"""# Part 3 - Architecture Detective (2 Points)

In this part of the homework you will play the architecture detective for five mystery models. These models are/were popular models of NLP. We standardized their state dicts. Your task is to determine the following information about each model looking at their state_dicts.

It might look like:
```python
{
  "vocab_size": 20, # int
  "d_model": 12, # int
  "ffn_size": 32, # int
  "num_hidden_layers": 2, # int
  "norm_type": "rmsnorm", # "layernorm" or "rmsnorm"
  "attention_type": "mha", # "mha" (for multi-head attention) or "gqa" (for grouped-query attention)
  "is_gated": False,
  "is_tied_embeddings": False,
  "decoder_only": False,
}
```

If you are curious and want to go the extra mile you can also try to match the model architectures to their source.

## General Hints:
At this point in the class you should be familiar with most of these terms and be able to identify them by looking at the state_dicts.

- `is_tied_embeddings`: for small models, we might tie (share) the input and output embeddings. [True or False]
- `is_gated`: indicates that the feed-forward network includes an additional linear layer
  to gate activations (e.g., GEGLU, SwiGLU). [True or False]
- `decoder_only`: True if the model follows a decoder-only architecture, False if it is an encoder-decoder model.


  The models are listed below:
  - 0: https://huggingface.co/r-three/mystery-model-0/blob/main/model.safetensors
  - 1: https://huggingface.co/r-three/mystery-model-1/blob/main/model.safetensors
  - 2: https://huggingface.co/r-three/mystery-model-2/blob/main/model.safetensors
  - 3: https://huggingface.co/r-three/mystery-model-3/blob/main/model.safetensors
  - 4: https://huggingface.co/r-three/mystery-model-4/blob/main/model.safetensors
"""

default_model_dct = {
    "vocab_size": 0,
    "d_model": 0,
    "ffn_size": 0,
    "num_hidden_layers": 0,
    # "layernorm or rmsnorm"
    "norm_type": "",
    # attention_type should be "mha (multi-head) or gqa (grouped-query)",
    "attention_type":  "",
    # is_gated, is_tied_embeddings, and decoder_only should be boolean
    "is_gated": None,
    "is_tied_embeddings": None,
    "decoder_only": None,
}

def type_check(model_dct):
    assert all([k in default_model_dct.keys() for k in model_dct.keys()]), "Don't add additional keys"
    assert all([k in model_dct.keys() for k in default_model_dct.keys()]), "Don't miss any keys"
    assert isinstance(model_dct["vocab_size"], int), "vocab_size should be an integer"
    assert isinstance(model_dct["d_model"], int), "d_model should be an integer"
    assert isinstance(model_dct["ffn_size"], int), "ffn_size should be an integer"
    assert isinstance(model_dct["num_hidden_layers"], int), "num_hidden_layers should be an integer"
    assert model_dct["norm_type"] in ["layernorm", "rmsnorm"]
    assert model_dct["attention_type"] in ["mha", "gqa"]
    assert isinstance(model_dct["is_gated"], bool), "is_gated should be a boolean"
    assert isinstance(model_dct["is_tied_embeddings"], bool), "is_tied_embeddings should be a boolean"
    assert isinstance(model_dct["decoder_only"], bool), "decoder_only should be a boolean"

def mystery_model_0():
    res = default_model_dct
    ########################### YOUR CODE ###################################
    ## TODO:
    # model.embed_tokens.weight: [128256, 2048] => vocab_size = 128256, d_model = 2048
    # model.layers.0.ffn.w1.weight: [2048, 8192] => d_ff = 8192
    # model.layers.0 ~ model.layers.15 => 16 hidden layers
    # model.layers.0.input_norm.weight: [2048] => No bias so rmsnorm
    # model.layers.0.self_attn.k_proj.weight: [512, 2048] => d_k = 512
    # model.layers.0.self_attn.q_proj.weight: [2048, 2048] => d_q = 2048
    # d_q > d_k => gqa

    res.update({
        'vocab_size': 128256,
        'd_model': 2048,
        'ffn_size': 8192,
        'num_hidden_layers': 16,
        'norm_type': 'rmsnorm',
        'attention_type': 'gqa',
        'is_gated': True,
        'is_tied_embeddings': True,
        'decoder_only': True
    })
    #########################################################################
    return res

def mystery_model_1():
    res = default_model_dct
    ########################### YOUR CODE ###################################
    ## TODO:
    # model.embed_tokens.weight: [262144, 1152] => vocab_size = 262144, d_model = 1152
    # model.layers.0.ffn.w1.weight: [1152, 6912] => d_ff = 6912
    # model.layers.0 ~ model.layers.25 => 26 hidden layers
    # model.layers.0.input_norm.weight: [1152] => No bias so rmsnorm
    # model.layers.0.self_attn.k_proj.weight: [256, 1152] => d_k = 256
    # model.layers.0.self_attn.q_proj.weight: [1024, 1152] => d_q = 1024
    # d_q > d_k => gqa

    res.update({
        'vocab_size': 262144,
        'd_model': 1152,
        'ffn_size': 6912,
        'num_hidden_layers': 26,
        'norm_type': 'rmsnorm',
        'attention_type': 'gqa',
        'is_gated': True,
        'is_tied_embeddings': True,
        'decoder_only': True
    })
    #########################################################################
    return res


def mystery_model_2():
    res = default_model_dct
    ########################### YOUR CODE ###################################
    ## TODO:
    res.update({
        'vocab_size': 50304,
        'd_model': 512,
        'ffn_size': 2048,
        'num_hidden_layers': 6,
        'norm_type': 'layernorm',
        'attention_type': 'mha',
        'is_gated': False,
        'is_tied_embeddings': False,
        'decoder_only': True
    })
    #########################################################################
    return res


def mystery_model_3():
    res = default_model_dct
    ########################### YOUR CODE ###################################
    ## TODO:
    res.update({
        'vocab_size': 32128,
        'd_model': 768,
        'ffn_size': 3072,
        'num_hidden_layers': 12,
        'norm_type': 'rmsnorm',
        'attention_type': 'mha',
        'is_gated': False,
        'is_tied_embeddings': True,
        'decoder_only': False
    })
    #########################################################################
    return res


def mystery_model_4():
    res = default_model_dct
    ########################### YOUR CODE ###################################
    ## TODO:
    res.update({
        'vocab_size': 32768,
        'd_model': 4096,
        'ffn_size': 14336,
        'num_hidden_layers': 32,
        'norm_type': 'rmsnorm',
        'attention_type': 'gqa',
        'is_gated': True,
        'is_tied_embeddings': False,
        'decoder_only': True
    })
    #########################################################################
    return res

if __name__ == "__main__":
    type_check(mystery_model_0())
    type_check(mystery_model_1())
    type_check(mystery_model_2())
    type_check(mystery_model_3())
    type_check(mystery_model_4())
