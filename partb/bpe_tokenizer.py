from pathlib import Path
UNK = '<|UNK|>'
class Token:
    leaf: bool
    val: str
    def __init__(self, val=None, left=None, right=None):
        self.leaf = val is not None
        if self.leaf:
            self.val = val
        else:
            self.left = left
            self.right = right

    def __eq__(self, other):
        if self.leaf and other.leaf:
            return self.val == other.val
        elif not self.leaf and not other.leaf:
            return self.left == other.left and self.right == other.right
        return False

    def __hash__(self):
        if self.leaf:
            return hash(self.val)
        else:
            return hash((self.left, self.right))

    def expand(self):
        if self.leaf:
            return [self]
        else:
            return self.left.expand() + self.right.expand()

class BPETokenizer:
    vocab_size: int
    vocab: list[Token]
    mapping: dict[Token, int]
    base_size: int

    def __init__(self, vocab_size=1000, special_tokens=None):
        self.vocab_size = vocab_size
        self.vocab = []

    def train(self, corpus):
        base_vocab = {UNK: 0}
        for s in corpus:
            for c in s:
                base_vocab[c] = base_vocab.get(c, 0) + 1
        self.base_size = len(base_vocab)
        self.vocab.extend([Token(c) for c, _ in base_vocab.items()])
        mapping = {}
        for i in range(len(self.vocab)):
            mapping[self.vocab[i]] = i
        
        word_freqs = {}
        pairs = {}
        pair_locations = {}
        for s in corpus:
            for w in s.split():
                w = tuple(Token(c) for c in w)
                word_freqs[w] = word_freqs.get(w, 0) + 1
                for i in range(len(w)-1):
                    p = Token(left=w[i], right=w[i+1])
                    pairs[p] = pairs.get(p, 0) + 1
                    if p not in pair_locations:
                        pair_locations[p] = set()
                    pair_locations[p].add(w)

        while len(self.vocab) < self.vocab_size:
            freq_pair, _ = max(pairs.items(), key=lambda x: x[1])
            for w in set(pair_locations[freq_pair]):
                for i in range(len(w) - 1):
                    p = Token(left=w[i], right=w[i+1])
                    pairs[p] -= word_freqs[w]
                    pair_locations[p].discard(w)

                new_w = []
                i = 0
                while i < len(w):
                    if i < len(w) - 1 and w[i] == freq_pair.left and w[i+1] == freq_pair.right:
                        new_w.append(freq_pair)
                        i += 2
                    else:
                        new_w.append(w[i])
                        i += 1
                new_w = tuple(new_w)

                for i in range(len(new_w) - 1):
                    p = Token(left=new_w[i], right=new_w[i+1])
                    pairs[p] = pairs.get(p, 0) + word_freqs[w]
                    if p not in pair_locations:
                        pair_locations[p] = set()
                    pair_locations[p].add(new_w)
                
                word_freqs[new_w] = word_freqs[w]
                del word_freqs[w]
            
            self.vocab.append(freq_pair)
            mapping[freq_pair] = len(self.vocab) - 1
            del pairs[freq_pair]
            del pair_locations[freq_pair]
        
        self.mapping = {t: mapping[t] for t in self.vocab}
    
    def encode(self, text):
        words = text.split()
        encoded_words = []
        for idx in range(len(words)):
            w_ = words[idx]
            w = tuple(Token(c) for c in w_)
            for i in range(len(w)):
                if w[i] not in self.mapping:
                    w[i] = Token(UNK)
            modified = True
            while modified:
                modified = False
                pairs = []
                for i in range(len(w) - 1):
                    p = Token(left=w[i], right=w[i+1])
                    if p in self.mapping:
                        pairs.append(p)
                        modified = True
                if not modified:
                    break
                top_pair = min(pairs, key=lambda x: self.mapping[x])
                new_w = []
                i = 0
                while i < len(w):
                    if i < len(w) - 1 and w[i] == top_pair.left and w[i+1] == top_pair.right:
                        new_w.append(top_pair)
                        i += 2
                    else:
                        new_w.append(w[i])
                        i += 1
                w = tuple(new_w)
            encoded_words.append(w)

        encoded_sentence = []
        for w in encoded_words:
            encoded_sentence.extend([self.mapping[t] for t in w])
            encoded_sentence.append(self.mapping[Token(' ')])
        encoded_sentence = encoded_sentence[:-1]
        return encoded_sentence
    
    def decode(self, token_ids):
        expanded_tokens = []
        for t in token_ids:
            expanded_tokens.extend(self.vocab[t].expand())
        return ''.join(t.val for t in expanded_tokens)

    def save(self, filepath):
        dir = Path(filepath)
        file = dir / "tokens.txt"
        with open(file, 'w') as f:
            f.write(str(self.base_size) + '\n')
            for t in self.vocab:
                if t.leaf:
                    f.write(t.val + '\n')
                else:
                    f.write(str(self.mapping[t.left]) + ' ' + str(self.mapping[t.right]) + '\n')

    def load(self, filepath):
        dir = Path(filepath)
        file = dir / "tokens.txt"
        with open(file, 'r') as f:
            lines = f.read().splitlines()
            self.base_size = int(lines[0])
            self.vocab = []
            self.mapping = {}
            for i in range(1, len(lines)):
                line = lines[i]
                if i <= self.base_size:
                    token = Token(val=line)
                else:
                    left_idx, right_idx = map(int, line.split())
                    token = Token(left=self.vocab[left_idx], right=self.vocab[right_idx])
                
                self.vocab.append(token)
                self.mapping[token] = i - 1
            
    def get_vocab_size(self):
        return self.vocab_size
    
    def get_unk_id(self):
        return 0
