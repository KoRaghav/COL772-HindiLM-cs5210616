import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List
from dataclasses import dataclass

@dataclass
class AttentionWeights:
    W_Q: torch.Tensor
    W_K: torch.Tensor
    W_V: torch.Tensor

class Attention(nn.Module):
    def __init__(self, d_model: int, mode: str):
        super().__init__()
        self.d_model = d_model
        self.W_Q: torch.Tensor
        self.W_K: torch.Tensor
        self.W_V: torch.Tensor
    
    def set_weights(self, weights: AttentionWeights):
        self.W_Q = weights.W_Q
        self.W_K = weights.W_K
        self.W_V = weights.W_V

    def forward(self, input: torch.Tensor):
        Q = torch.matmul(input, self.W_Q.t())
        K = torch.matmul(input, self.W_K.t())
        V = torch.matmul(input, self.W_V.t())
        S = torch.matmul(Q, K.reshape(K.shape[0], K.shape[2], -1))
        S /= self.d_model
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
    def __init__(self, d_model: int, mode: str, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        self.layernorm_1 = nn.LayerNorm(self.d_model)
        self.layernorm_2 = nn.LayerNorm(self.d_model)

        self.heads = [Attention(d_model, mode) for _ in range(n_heads)]
        self.W_O: torch.Tensor

        self.W_up: torch.Tensor
        self.b_up: torch.Tensor
        self.W_down: torch.Tensor
        self.b_down: torch.Tensor

    def set_weights(self, weights: TransformerBlockWeights):
        with torch.no_grad():
            self.layernorm_1.weight.copy_(weights.gamma_1)
            self.layernorm_1.bias.copy_(weights.beta_1)
            self.layernorm_2.weight.copy_(weights.gamma_2)
            self.layernorm_2.bias.copy_(weights.beta_2)
        
        for i in range(self.n_heads):
            self.heads[i].set_weights(weights.attention_weights[i])
        self.W_O = weights.W_O

        self.W_up = weights.W_up
        self.b_up = weights.b_up
        self.W_down = weights.W_down
        self.b_down = weights.b_down

    def forward(self, x):
        u = self.layernorm_1(x)
        heads = [att(u) for att in self.heads]
        heads_cat = torch.cat(heads, axis=2)
        attn = torch.matmul(heads_cat, self.W_O)
        z1 = x + attn
        v1 = self.layernorm_2(z1)
        v2 = torch.matmul(v1, self.W_up) + self.b_up
        v3 = F.gelu(v2)
        v4 = torch.matmul(v3, self.W_down) + self.b_down
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
        self.D_HEAD = self.D_MODEL / self.N_HEADS
        self.N_LAYERS = config["n_layers"]
        self.VOCAB_SIZE = config["vocab_size"]
        self.MODE = config["mode"]
        self.TAU = config.get("tau")

        self.W_vocab: torch.Tensor
        self.W_devocab: torch.Tensor
        self.transformer_blocks = [TransformerBlock(self.D_MODEL, self.MODE, self.N_HEADS) for _ in range(self.N_LAYERS)]
        self.layernorm_final = nn.LayerNorm(self.D_MODEL)

    def set_weights(self, weights: Dict[str, Any]):
        """
        Set the model's weights based on the provided dictionary.
        The weights dictionary will contain all necessary parameters to initialize the model's layers.
        You should ensure that the weights are correctly assigned to the corresponding layers in your model.

        Parameters:
            - weights: A dictionary containing the model's weights. The structure of this dictionary will depend on how you design your model.
        """
        self.W_vocab = weights["W_vocab"]
        self.W_devocab = weights["W_devocab"]

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
        batch_size, l = input_ids.shape

        embed_layer = nn.Embedding(self.VOCAB_SIZE, self.D_MODEL, _weight=torch.transpose(self.W_vocab, 0, 1))
        embedded = embed_layer(input_ids)

        pos_embed_even = torch.sin(torch.arange(l).unsqueeze(1) / (10000 ** (torch.arange(0, self.D_MODEL, 2).unsqueeze(0).expand(l, -1) / self.D_MODEL)))
        pos_embed_odd = torch.cos(torch.arange(l).unsqueeze(1) / (10000 ** (torch.arange(1, self.D_MODEL, 2).unsqueeze(0).expand(l, -1) / self.D_MODEL)))
        pos_embed = torch.zeros(l, self.D_MODEL)
        pos_embed[:, 0::2] = pos_embed_even
        pos_embed[:, 1::2] = pos_embed_odd

        x = embedded + pos_embed.unsqueeze(0)

        for l in range(self.N_LAYERS):
            x = self.transformer_blocks[l](x)

        x_final = self.layernorm_final(x)

        raise NotImplementedError("Implement forward as described in assignment document")



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
    max_len = max(len(ids) for ids in batch["input_ids"])
    padded_inp = [F.pad(input_id, (0, max_len-len(input_id)), value=PAD_ID) for input_id in batch["input_ids"]]
    padded_attn = [F.pad(attn, (0, max_len-len(attn)), value=PAD_ID) for attn in batch["attention_mask"]]
    stacked_inp = torch.stack(padded_inp)
    stacked_attn = torch.stack(padded_attn)
    return {"input_ids": stacked_inp, "attention_mask": stacked_attn}

if __name__ == "__main__":
    import pathlib
    import time
    from COL772.parta.check import read_config, read_data, read_weights, match

    def run_model(model: nn.Module, input_ids: List[torch.Tensor], vocab_size: int) -> List[Dict[str, torch.Tensor]]:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        model.eval()
        model.to(device)

        outputs = []
        bsz = 16
        for st in range(0, len(input_ids), bsz):
            en = min(st + bsz, len(input_ids))
            batch = {
                "input_ids": [input_ids[i] for i in range(st, en)],
                "attention_mask": [torch.ones_like(input_ids[i]) for i in range(st, en)]
            }
            padded_batch = collate_fn(batch)
            padded_batch = {k: v.to(device) for k, v in padded_batch.items()}
            with torch.no_grad():
                logits = model(input_ids=padded_batch["input_ids"], attention_mask=padded_batch["attention_mask"])
            logits = logits.cpu()
            for i in range(en - st):
                outputs.append({
                    "logits": logits[i][:len(batch["input_ids"][i])]
                })

        return outputs

    config = read_config(pathlib.Path("/content/COL772/parta/data/case1/config.json"))
    input_ids, gold_outputs = read_data(pathlib.Path("/content/COL772/parta/data/case1/model_outputs"))
    state_dict = read_weights(pathlib.Path("/content/COL772/parta/data/case1/model_weights.pth"))

    model = load_model(config=config, weights=state_dict)
    st = time.time()
    outputs = run_model(model=model, input_ids=input_ids, vocab_size=config["vocab_size"])
    en = time.time()
    match(gold_outputs=gold_outputs, model_outputs=outputs)
    print(f"Inference time: {en - st} seconds")