import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from math import sqrt
from typing import Any, Dict, List
from dataclasses import dataclass

@dataclass
class AttentionWeights:
    W_Q: torch.Tensor
    W_K: torch.Tensor
    W_V: torch.Tensor

class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_head: int, mode: str, tau: float):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.mode = mode
        self.tau = tau
        
        self.W_Q = nn.Parameter(torch.randn(n_heads * d_head, d_model) / sqrt(d_model))
        self.W_K = nn.Parameter(torch.randn(n_heads * d_head, d_model) / sqrt(d_model))
        self.W_V = nn.Parameter(torch.randn(n_heads * d_head, d_model) / sqrt(d_model))

    def forward(self, input: torch.Tensor, attention_mask: torch.Tensor):
        B, L, D = input.shape
        q = F.linear(input, self.W_Q)
        k = F.linear(input, self.W_K)
        v = F.linear(input, self.W_V)
        q = q.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        S = torch.matmul(q, k.transpose(-2, -1)) / sqrt(self.d_head)
        if self.mode == "tanh-clipped":
            S = self.tau * torch.tanh(S)
        upper_triangle = torch.triu(torch.ones(L, L, device=input.device), diagonal=1).bool()
        S = S.masked_fill(upper_triangle.unsqueeze(0).unsqueeze(0), -1e9)
        S = S.masked_fill(attention_mask.unsqueeze(1).unsqueeze(2) == 0, -1e9)
        attn = F.softmax(S, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, L, -1)
        return out

@dataclass
class TransformerBlockWeights:
    attention_weights: list[AttentionWeights]
    beta_1: torch.Tensor
    beta_2: torch.Tensor
    gamma_1: torch.Tensor
    gamma_2: torch.Tensor
    W_O: torch.Tensor
    W_up: torch.Tensor
    W_down: torch.Tensor
    b_up: torch.Tensor
    b_down: torch.Tensor

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, mode: str, tau: float, n_heads: int, d_head: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head

        self.layernorm_1 = nn.LayerNorm(self.d_model, elementwise_affine=True)
        self.layernorm_2 = nn.LayerNorm(self.d_model, elementwise_affine=True)

        self.attention = Attention(d_model, n_heads, d_head, mode, tau)
        self.W_O = nn.Parameter(torch.randn(d_model, n_heads * d_head) / sqrt(n_heads * d_head))

        self.W_up = nn.Parameter(torch.randn(d_model, 4 * d_model) / sqrt(d_model))
        self.b_up = nn.Parameter(torch.zeros(4 * d_model))
        self.W_down = nn.Parameter(torch.randn(4 * d_model, d_model) / sqrt(4 * d_model))
        self.b_down = nn.Parameter(torch.zeros(d_model))

    def forward(self, x, attention_mask):
        u = self.layernorm_1(x)
        attn = self.attention(u, attention_mask)
        attn = attn @ self.W_O.t()
        z1 = x + attn
        v1 = self.layernorm_2(z1)
        v2 = v1 @ self.W_up + self.b_up
        v3 = F.gelu(v2)
        v4 = v3 @ self.W_down + self.b_down
        z2 = z1 + v4
        return z2

class LanguageModel(nn.Module):
    """
    This is a stub class for the assignment.
    Feel free to change the function signatures (including that of __init__, forward) as you need them.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Build the LanguageModel based on the config.
        """
        super().__init__()

        self.D_MODEL = config["d_model"]
        self.N_HEADS = config["n_heads"]
        self.D_HEAD = self.D_MODEL // self.N_HEADS
        self.N_LAYERS = config["n_layers"]
        self.VOCAB_SIZE = config["vocab_size"]
        self.MODE = config["mode"]
        self.TAU = config.get("tau")

        self.W_vocab = nn.Parameter(torch.randn(self.D_MODEL, self.VOCAB_SIZE) / sqrt(self.D_MODEL))
        self.W_devocab = nn.Parameter(torch.randn(self.D_MODEL, self.VOCAB_SIZE) / sqrt(self.D_MODEL))
        self.transformer_blocks = nn.ModuleList([TransformerBlock(self.D_MODEL, self.MODE, self.TAU, self.N_HEADS, self.D_HEAD) for _ in range(self.N_LAYERS)])
        self.layernorm_final = nn.LayerNorm(self.D_MODEL, elementwise_affine=True)
        
        self.max_seq_len = 512 
        self.register_buffer('pos_embed', self._get_pos_embed(self.max_seq_len, self.D_MODEL))

    def _get_pos_embed(self, L, D):
        pe = torch.zeros(L, D)
        position = torch.arange(0, L, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, D, 2).float() * (-torch.log(torch.tensor(10000.0)) / D))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def set_weights(self, weights: Dict[str, Any]):
        """
        Set the model's weights based on the provided dictionary.
        The weights dictionary will contain all necessary parameters to initialize the model's layers.
        You should ensure that the weights are correctly assigned to the corresponding layers in your model.

        Parameters:
            - weights: A dictionary containing the model's weights. The structure of this dictionary will depend on how you design your model.
        """
        self.W_vocab = nn.Parameter(weights["W_vocab"])
        self.W_devocab = nn.Parameter(weights["W_devocab"])

        for l in range(1, self.N_LAYERS+1):
            attention_weights = []
            for k in range(1, self.N_HEADS+1):
                attention_weights.append(AttentionWeights(weights[f"W_{l}_Q_{k}"], weights[f"W_{l}_K_{k}"], weights[f"W_{l}_V_{k}"]))
            transformer_weights = TransformerBlockWeights(
                attention_weights,
                weights[f"beta_{l}_1"],
                weights[f"beta_{l}_2"],
                weights[f"gamma_{l}_1"],
                weights[f"gamma_{l}_2"],
                weights[f"W_{l}_O"],
                weights[f"W_{l}_up"],
                weights[f"W_{l}_down"],
                weights[f"b_{l}_up"],
                weights[f"b_{l}_down"],
            )
            self.transformer_blocks[l-1].set_weights(transformer_weights)
        
        with torch.no_grad():
            self.layernorm_final.weight.copy_(weights["gamma_final"])
            self.layernorm_final.bias.copy_(weights["beta_final"])
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Implement the forward pass of the model. The output should be a tensor of shape (T, |Vocab|).

        Parameters:
            - input_ids: A tensor of shape (batch_size, sequence_len) containing token IDs.
            - attention_mask: A tensor of shape (batch_size, sequence_len) containing 1s for valid tokens and 0s for padding.

        Returns:
            - A tensor of shape (batch_size, sequence_len, vocab_size) containing the logits for each token in the vocabulary.
            Logits are the raw, unnormalized scores output by the model, which can be converted to probabilities using a softmax function.
        """
        _, L = input_ids.shape

        embedded = F.embedding(input_ids, self.W_vocab.t())

        if L > self.pos_embed.size(1):
             self.register_buffer('pos_embed', self._get_pos_embed(L, self.D_MODEL))
        
        x = embedded + self.pos_embed[:, :L, :]

        for l in range(self.N_LAYERS):
            x = self.transformer_blocks[l](x, attention_mask)

        x_final = self.layernorm_final(x)

        logits = x_final @ self.W_devocab

        return logits



def load_model(config: Dict[str, Any], weights: Dict[str, Any]):
    """
    This is a sample code. Replace with your own.
    However, DO NOT CHANGE THE SIGNATURE OF THIS FUNCTION.
    Ensure that the function inputs config and weights and outputs a nn.Module derived object.
    """

    model = LanguageModel(config)
    model.set_weights(weights)

    return model


def collate_fn(batch: Dict[str, List[torch.tensor]]) -> Dict[str, torch.Tensor]:
    """
    This is a sample code. Replace with your own.
    However, DO NOT CHANGE THE SIGNATURE OF THIS FUNCTION.
    Ensure that the function takes in a batch of data and outputs a dictionary of tensors ready to be fed into the model.
    """
    PAD_ID = 0  # Assume 0 is the padding token ID
    input_ids = pad_sequence(batch["input_ids"], batch_first=True, padding_value=PAD_ID)
    attention_mask = pad_sequence(batch["attention_mask"], batch_first=True)
    return {"input_ids": input_ids, "attention_mask": attention_mask}