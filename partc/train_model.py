# YOUR TOKENIZER AND MODEL from PART A AND PART B RESPECTIVELY
# If you wish to change their code, please do so in their respective files under parta/ and partb/ directories.
from partb.bpe_tokenizer import BPETokenizer
from parta.model import LanguageModel
# You can also create additional files in this directory and import them here if needed.
# For example, the line below import a dummy function from utils.py file.
from .utils import collate_fn
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# You can structure your code as you see fit as long as the CLI works as specified.
# Finally, treat this as your FINAL MODEL TRAINING SCRIPT. Do not perform hyperparameter tuning here.
# You can create separate scripts for hyperparameter tuning if needed.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_EPOCHS = 100
BATCH_SIZE = 64
MAX_LENGTH = 128

LIMIT_TRAIN = 100000

config = {
    "d_model": 128,
    "n_heads": 4,
    "d_head": 32,
    "n_layers": 6,
    "mode": "standard"
}

class HindiDataset(Dataset):
    def __init__(self, corpus, tokenizer, max_length=MAX_LENGTH):
        self.data = []
        for text in corpus:
            seq = tokenizer.encode(text)
            if len(seq) > 1:
                if len(seq) > max_length + 1:
                    seq = seq[:max_length + 1]
                self.data.append(torch.tensor(seq))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def main(args):
    corpus = []
    with open(args.train_path, 'r', encoding='utf-8') as f:
        for line in f:
            corpus.append(line.strip())

    tokenizer = BPETokenizer()
    tokenizer.load(args.tokenizer_path)
    config['vocab_size'] = tokenizer.get_vocab_size()
    print(f"Tokenizer loaded from {args.tokenizer_path}.")

    dataset = HindiDataset(corpus, tokenizer)

    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn,
        pin_memory=(device.type == 'cuda'),
        num_workers=2
    )

    model = LanguageModel(config).to(device)

    lossCriterion = nn.CrossEntropyLoss(ignore_index=0).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
    for epoch in range(1, NUM_EPOCHS+1):
        total_loss = 0

        for input_ids, targets, mask in loader:
            input_ids, targets, mask = input_ids.to(device), targets.to(device), mask.to(device)
            optimizer.zero_grad()
            logits = model(input_ids, mask)
            loss = lossCriterion(logits.reshape(-1, config['vocab_size']), targets.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch}: loss: {total_loss}")
        torch.save(model.state_dict(), args.output_model_path)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train a model on the given dataset.')
    parser.add_argument('--train_path', type=str, required=True, help='Path to the train dataset')
    parser.add_argument('--valid_path', type=str, required=True, help='Path to the valid dataset')
    parser.add_argument('--tokenizer_path', type=str, required=True, help='Path to the tokenizer')
    parser.add_argument('--output_model_path', type=str, default='checkpoints', help='Directory to save checkpoints')

    args = parser.parse_args()
    main(args)
