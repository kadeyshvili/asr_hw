import re
from string import ascii_lowercase

import torch
from collections import defaultdict
from pyctcdecode import BeamSearchDecoderCTC, Alphabet, LanguageModel
import kenlm
import os

# TODO add CTC decode
# TODO add BPE, LM, Beam Search support
# Note: think about metrics and encoder
# The design can be remarkably improved
# to calculate stuff more efficiently and prettier


class CTCTextEncoder:
    EMPTY_TOK = ""
    def __init__(self, use_lm=False, vocab_path=None, model_path=None,lm_path=None, alphabet=None, **kwargs):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """

        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")

        self.alphabet = alphabet
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)

        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}
        if not use_lm:
            language_model = None
        else:
            with open(vocab_path) as f:
                unigram_list = [t.lower() for t in f.read().strip().split("\n")]
            alpha = 0.5
            beta = 1.5
            lm_path = lm_path
            if not os.path.exists(lm_path):
                with open(model_path, 'r') as f_upper:
                    print(f_upper)
                    with open(lm_path, 'w') as f_lower:
                        for line in f_upper:
                            f_lower.write(line.lower())

            kenlm_model = kenlm.Model(lm_path)

            language_model = LanguageModel(kenlm_model=kenlm_model, unigrams=unigram_list, alpha=alpha, beta=beta)
        self.decoder = BeamSearchDecoderCTC(alphabet=Alphabet(labels=self.vocab, is_bpe=False), language_model=language_model)

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            return torch.Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
        except KeyError:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
            )

    def decode(self, inds) -> str:
        """
        Raw decoding without CTC.
        Used to validate the CTC decoding implementation.

        Args:
            inds (list): list of tokens.
        Returns:
            raw_text (str): raw text with empty tokens and repetitions.
        """
        return "".join([self.ind2char[int(ind)] for ind in inds]).strip()

    def ctc_decode(self, inds) -> str:
        decoded = []
        last_char = self.EMPTY_TOK
        for ind in inds:
            cur_char = self.ind2char[ind]
            if cur_char == self.EMPTY_TOK:
                continue
            if cur_char != last_char:
                decoded.append(cur_char)
            last_char = cur_char
        return ''.join(decoded)



    def expand_and_merge_path(self, dp, next_token_probs):
        new_dp = defaultdict(float)
        for ind, next_token_prob in enumerate(next_token_probs):
            cur_char = self.ind2char[ind]
            for (prefix, last_char), v in dp.items():
                if last_char == cur_char:
                    new_prefix = prefix
                else:
                    if cur_char != self.EMPTY_TOK:
                        new_prefix = prefix + cur_char
                    else:
                        new_prefix = prefix
                new_dp[(new_prefix, cur_char)] += v * next_token_prob
        return new_dp

    def truncate_paths(self, dp, beam_size):
        return dict(sorted(list(dp.items()), key = lambda x : -x[1])[:beam_size])

    def ctc_beam_search(self, probs, beam_size=10):
        dp = {('', self.EMPTY_TOK) : 1.0, }
        for prob in probs:
            dp = self.expand_and_merge_path(dp, prob)
            dp = self.truncate_paths(dp, beam_size)
        dp = [(prefix, proba) for (prefix, _), proba in sorted(dp.items(), key=lambda x : -x[1])]
        return dp
 
    def ctc_beam_search_module(self, probs, beam_width=10):
        return [self.decoder.decode(probs, beam_width)]
    
    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
