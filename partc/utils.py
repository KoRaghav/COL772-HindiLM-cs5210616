from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    input_ids = [s[:-1] for s in batch]
    target_ids = [s[1:] for s in batch]
    input_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    target_padded = pad_sequence(target_ids, batch_first=True, padding_value=0)
    attention_mask = (input_padded != 0).float()
    return input_padded, target_padded, attention_mask