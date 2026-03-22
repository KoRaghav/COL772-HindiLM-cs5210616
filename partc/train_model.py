# YOUR TOKENIZER AND MODEL from PART A AND PART B RESPECTIVELY
# If you wish to change their code, please do so in their respective files under parta/ and partb/ directories.
from pathlib import Path
from partb.bpe_tokenizer import BPETokenizer
from parta.model import LanguageModel
# You can also create additional files in this directory and import them here if needed.
# For example, the line below import a dummy function from utils.py file.
from .utils import dummy_function  # Replace with actual utility functions as needed
import torch
import torch.nn as nn
# You can structure your code as you see fit as long as the CLI works as specified.
# Finally, treat this as your FINAL MODEL TRAINING SCRIPT. Do not perform hyperparameter tuning here.
# You can create separate scripts for hyperparameter tuning if needed.

NUM_EPOCHS = 10

def main(args):
    corpus = []
    with open(args.train_path, 'r', encoding='utf-8') as f:
        for line in f:
            corpus.append(line.strip())

    config = {
        "d_model": 128,
        "n_heads": 4,
        "d_head": 32,
        "n_layers": 8,
        "vocab_size": 50000,
        "mode": "standard"
    }
    tokenizer = BPETokenizer(config['vocab_size'])
    tokenizer.load(args.tokenizer_path)
    print(f"Tokenizer loaded from {args.tokenizer_path}.")

    encoded_corpus = [tokenizer.encode(s) for s in corpus]


    model = LanguageModel(config)

    loss = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train a model on the given dataset.')
    parser.add_argument('--train_path', type=str, required=True, help='Path to the train dataset')
    parser.add_argument('--valid_path', type=str, required=True, help='Path to the valid dataset')
    parser.add_argument('--tokenizer_path', type=str, required=True, help='Path to the tokenizer')
    parser.add_argument('--output_model_path', type=str, default='checkpoints', help='Directory to save checkpoints')

    args = parser.parse_args()
    main(args)
