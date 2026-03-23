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
    def __init__(self, d_model: int, mode: str, tau: float, d_head: int):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.W_Q = nn.Parameter(torch.randn(d_head, d_model))
        self.W_K = nn.Parameter(torch.randn(d_head, d_model))
        self.W_V = nn.Parameter(torch.randn(d_head, d_model))
        self.mode = mode
        self.tau = tau
    
    def set_weights(self, weights: AttentionWeights):
        self.W_Q = nn.Parameter(weights.W_Q)
        self.W_K = nn.Parameter(weights.W_K)
        self.W_V = nn.Parameter(weights.W_V)

    def forward(self, input: torch.Tensor, attention_mask: torch.Tensor):
        Q = input @ self.W_Q.t()
        K = input @ self.W_K.t()
        V = input @ self.W_V.t()
        S = torch.bmm(Q, K.transpose(1, 2))
        S /= sqrt(self.d_head)
        if self.mode == "tanh-clipped":
            S = self.tau * torch.tanh(S)
        l = input.shape[1]
        M = torch.zeros(l, l, dtype=torch.float32)
        upper_triangle = torch.triu(torch.ones(l, l), diagonal=1).bool()
        M = M.masked_fill(upper_triangle, -torch.inf).to(input.device)
        S = S + M
        attention_mask = attention_mask.unsqueeze(1)
        S = S.masked_fill(attention_mask == 0, -torch.inf)
        Attn = F.softmax(S, dim=-1)
        return torch.matmul(Attn, V)

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

        self.heads = nn.ModuleList([Attention(d_model, mode, tau, d_head) for _ in range(n_heads)])
        self.W_O = nn.Parameter(torch.randn(d_model, d_model))

        self.W_up = nn.Parameter(torch.randn(d_model, n_heads*d_model))
        self.b_up = nn.Parameter(torch.zeros(n_heads*d_model))
        self.W_down = nn.Parameter(torch.randn(n_heads*d_model, d_model))
        self.b_down = nn.Parameter(torch.zeros(d_model))

    def set_weights(self, weights: TransformerBlockWeights):
        with torch.no_grad():
            self.layernorm_1.weight.copy_(weights.gamma_1)
            self.layernorm_1.bias.copy_(weights.beta_1)
            self.layernorm_2.weight.copy_(weights.gamma_2)
            self.layernorm_2.bias.copy_(weights.beta_2)
        
        for i in range(self.n_heads):
            self.heads[i].set_weights(weights.attention_weights[i])
        self.W_O = nn.Parameter(weights.W_O)

        self.W_up = nn.Parameter(weights.W_up)
        self.b_up = nn.Parameter(weights.b_up)
        self.W_down = nn.Parameter(weights.W_down)
        self.b_down = nn.Parameter(weights.b_down)

    def forward(self, x, attention_mask):
        u = self.layernorm_1(x)
        heads = [att(u, attention_mask) for att in self.heads]
        heads_cat = torch.cat(heads, axis=2)
        attn = heads_cat @ self.W_O.t()
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

        self.W_vocab = nn.Parameter(torch.randn(self.D_MODEL, self.VOCAB_SIZE))
        self.W_devocab = nn.Parameter(torch.randn(self.D_MODEL, self.VOCAB_SIZE))
        self.transformer_blocks = nn.ModuleList([TransformerBlock(self.D_MODEL, self.MODE, self.TAU, self.N_HEADS, self.D_HEAD) for _ in range(self.N_LAYERS)])
        self.layernorm_final = nn.LayerNorm(self.D_MODEL, elementwise_affine=True)

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

        pos_embed_even = torch.sin(torch.arange(L, dtype=torch.float32).unsqueeze(1) / (10000 ** (torch.arange(0, self.D_MODEL, 2, dtype=torch.float32).unsqueeze(0).expand(L, -1) / self.D_MODEL)))
        pos_embed_odd = torch.cos(torch.arange(L, dtype=torch.float32).unsqueeze(1) / (10000 ** (torch.arange(0, self.D_MODEL, 2, dtype=torch.float32).unsqueeze(0).expand(L, -1) / self.D_MODEL)))
        pos_embed = torch.zeros(L, self.D_MODEL)
        pos_embed[:, 0::2] = pos_embed_even
        pos_embed[:, 1::2] = pos_embed_odd
        pos_embed = pos_embed.to(input_ids.device)

        x = embedded + pos_embed.unsqueeze(0)

        for l in range(self.N_LAYERS):
            x = self.transformer_blocks[l](x, attention_mask)

        x_final = self.layernorm_final(x)

        logits = x_final @ self.W_devocab

        probs = F.softmax(logits, dim=-1)

        return probs



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