import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List


class LanguageModel(nn.Module):
    """
    This is a stub class for the assignment.
    Feel free to change the function signatures (including that of __init__, forward) as you need them.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Build the LanguageModel based on the config.
        """

        self.D_MODEL = config["d_model"]
        self.N_HEADS = config["n_heads"]
        self.D_HEAD = self.D_MODEL / self.N_HEADS
        self.N_LAYERS = config["n_layers"]
        self.VOCAB_SIZE = config["vocab_size"]
        self.MODE = config["mode"]
        self.TAU = config.get("tau")

        self.weights = None
        super().__init__()

    def set_weights(self, weights: Dict[str, Any]):
        """
        Set the model's weights based on the provided dictionary.
        The weights dictionary will contain all necessary parameters to initialize the model's layers.
        You should ensure that the weights are correctly assigned to the corresponding layers in your model.

        Parameters:
            - weights: A dictionary containing the model's weights. The structure of this dictionary will depend on how you design your model.
        """
        self.weights = weights
        

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

        embed_layer = nn.Embedding(self.VOCAB_SIZE, self.D_MODEL, _weight=torch.transpose(self.weights["W_vocab"], 0, 1))
        embedded = embed_layer(input_ids)

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