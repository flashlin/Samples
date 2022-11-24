

class SimpleTokenizer(object):
    def __init__(self, tokenize_fn, bpe_path=default_bpe()):
        self.tokenize_fn = tokenize_fn
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = Path(bpe_path).read_text(encoding='utf8').split('\n')
        merges = merges[1:49152 - 256 - 2 + 1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v + '</w>' for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))
        vocab.extend([BOS_SYMBOL, EOS_SYMBOL, PAD_SYMBOL])

        # self.vocab_size = 49408
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.vocab_size = len(self.encoder)
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {BOS_SYMBOL: BOS_SYMBOL, EOS_SYMBOL: EOS_SYMBOL, PAD_SYMBOL: PAD_SYMBOL}
        # self.pattern = re.compile(
        #     r"""<start_of_text>|<end_of_text>""",
        #     re.IGNORECASE)
        self.bos_idx = self.encoder[BOS_SYMBOL]
        self.eos_idx = self.encoder[EOS_SYMBOL]
        self.padding_idx = self.encoder[PAD_SYMBOL]

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + '</w>',)
        pairs = get_pairs(word)

        if not pairs:
            return token + '</w>'

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text, add_start_end=False):
        bpe_tokens = []
        tokens = self.tokenize_fn(text)
        for token in tokens:
            token_text = ''.join(self.byte_encoder[b] for b in token.text.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token_text).split(' '))
        if add_start_end:
            bpe_tokens = [self.bos_idx] + bpe_tokens + [self.eos_idx]
        return bpe_tokens

    def decode(self, tokens, remove_start_end=True):
        if torch.is_tensor(tokens):
            tokens = tokens.tolist()

        if remove_start_end:
            tokens = [token for token in tokens if token not in (self.bos_idx, self.eos_idx)]
        text = ''.join([self.decoder[token] for token in tokens if token not in [self.padding_idx]])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', '')
        return text

    def tokenize(self, texts, context_length=256, truncate_text=False):
        if isinstance(texts, str):
            texts = [texts]

        all_tokens = [self.encode(text) for text in texts]
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                if truncate_text:
                    tokens = tokens[:context_length]
                else:
                    raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
            result[i, :len(tokens)] = torch.tensor(tokens)

        return result
