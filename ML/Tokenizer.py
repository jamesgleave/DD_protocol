"""
Used during testing to encode smiles as features
This is deprecated.
"""

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np


class DDTokenizer:
    def __init__(self, num_words, oov_token='<UNK>'):
        self.tokenizer = Tokenizer(num_words=num_words,
                                   oov_token=oov_token,
                                   filters='!"#$%&*+,-./:;<>?\\^_`{|}~\t\n',
                                   char_level=True,
                                   lower=False)
        self.has_trained = False

        self.pad_type = 'post'
        self.trunc_type = 'post'

        # The encoded data
        self.word_index = {}

    def fit(self, train_data):
        # Get max training sequence length
        print("Training Tokenizer...")
        self.tokenizer.fit_on_texts(train_data)
        self.has_trained = True
        print("Done training...")

        # Get our training data word index
        self.word_index = self.tokenizer.word_index

    def encode(self, data, use_padding=True, padding_size=None, normalize=False):
        # Encode training data sentences into sequences
        train_sequences = self.tokenizer.texts_to_sequences(data)

        # Get max training sequence length if there is none passed
        if padding_size is None:
            maxlen = max([len(x) for x in train_sequences])
        else:
            maxlen = padding_size

        if use_padding:
            train_sequences = pad_sequences(train_sequences, padding=self.pad_type,
                                            truncating=self.trunc_type, maxlen=maxlen)

        if normalize:
            train_sequences = np.multiply(1/len(self.tokenizer.word_index), train_sequences)

        return train_sequences

    def pad(self, data, padding_size=None):
        # Get max training sequence length if there is none passed
        if padding_size is None:
            padding_size = max([len(x) for x in data])

        padded_sequence = pad_sequences(data, padding=self.pad_type,
                                        truncating=self.trunc_type, maxlen=padding_size)

        return padded_sequence

    def decode(self, array):
        assert self.has_trained, "Train this tokenizer before decoding a string."
        return self.tokenizer.sequences_to_texts(array)

    def test(self, string):
        encoded = list(self.encode(string)[0])
        decoded = self.decode(self.encode(string))

        print("\nEncoding:")
        print("{original} -> {encoded}".format(original=string[0], encoded=encoded))
        print("\nDecoding:")
        print("{original} -> {encoded}".format(original=encoded, encoded=decoded[0].replace(" ", "")))

    def get_info(self):
        return self.tokenizer.index_word

