class Token:
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

class BPETokenizer:
    vocab_size: int
    vocab: list[Token]
    mapping: dict[Token, int]

    def __init__(self, vocab_size, special_tokens=None):
        self.vocab_size = vocab_size

    def train(self, corpus):
        base_vocab = {}
        for s in corpus:
            for c in s:
                base_vocab[c] = base_vocab.get(c, 0) + 1
        self.vocab = [Token(val=c) for c, _ in base_vocab.items()]
        mapping = {}
        for i in range(len(self.vocab)):
            mapping[self.vocab[i]] = i
        
        word_freqs = {}
        pairs = {}
        pair_locations = {}
        for s in corpus:
            for w in s.split():
                w = tuple(Token(val=c) for c in w)
                word_freqs[w] = word_freqs.get(w, 0) + 1
                for i in range(len(w)-1):
                    c1 = Token(val=w[i])
                    c2 = Token(val=w[i+1])
                    p = Token(left=c1, right=c2)
                    pairs[p] = pairs.get(p, 0) + 1
                    if p not in pair_locations:
                        pair_locations[p] = set()
                    pair_locations[p].add(mapping[w])

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
        
        self.mapping = {t: mapping[t] for t in self.vocab }
    
    def encode(self, text):
        raise NotImplementedError("Encoding method not implemented yet.")

    def decode(self, token_ids):
        raise NotImplementedError("Decoding method not implemented yet.")

    def save(self, filepath):
        raise NotImplementedError("Save method not implemented yet.")

    def load(self, filepath):
        raise NotImplementedError("Load method not implemented yet.")
    
    def get_vocab_size(self):
        raise NotImplementedError("Get vocab size method not implemented yet.")
    
    def get_unk_id(self):
        raise NotImplementedError("Get unk id method not implemented yet.")
