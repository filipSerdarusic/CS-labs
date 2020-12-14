import re
from collections import Counter
import numpy as np

class Dataset:

    def __init__(self, PATH, batch_size, sequence_length):
        self.path = PATH
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.counter = 0

    def preprocess(self, input_file):

        with open(input_file, "r") as f:
            data = f.read()
        
        # count and sort most frequent characters
        dat = list(re.sub(r'^[A-Za-z0-9! ‘ , . ` : ? ]+', '', data))
        self.sorted_chars = [char[0] for char in  Counter(dat).most_common()]
        
        # Number of characters in dataset (vocabulary size)
        self.vocab_size  = len(self.sorted_chars)

        self.char2id = dict(zip(
            self.sorted_chars, range(len(self.sorted_chars))))
    
        self.id2char = {k : v for v, k in self.char2id.items()}
        
        # convert the data to ids
        self.x = np.array(list(map(self.char2id.get, data)))


    def encode(self, sequence):
        # returns the sequence encoded as integers
        encoded = []
        for char in sequence:
            encoded.append(self.char2id[char])
        return encoded

    def decode(self, encoded_sequence):
        # returns the sequence decoded as letters
        decoded = ''
        for id in encoded_sequence:
            decoded += self.id2char[id]
        return decoded
    
    def create_minibatches(self):

        self.num_batches = int(len(self.x) / (self.batch_size * self.sequence_length))

        batch_len = self.batch_size * self.sequence_length
        
        self.batches_x = []
        self.batches_y = []

        for i in range(self.num_batches):
            batch_x = []
            batch_y = []
            for j in range(self.batch_size):

                batch_x.append(self.x[i*batch_len + j*self.sequence_length : i*batch_len + (j+1)*self.sequence_length])

                batch_y.append(self.x[i*batch_len + j*self.sequence_length + 1 : i*batch_len + (j+1)*self.sequence_length + 1])

            self.batches_x.append(batch_x)
            self.batches_y.append(batch_y)
    
        self.batches_x = np.array(self.batches_x)
        self.batches_y = np.array(self.batches_y)

        return self.batches_x, self.batches_y
    
    def next_minibatch(self):
        e = False
        if self.counter == self.num_batches:
            self.counter = 0
            e = True

        batch_x = self.batches_x[self.counter]
        batch_y = self.batches_y[self.counter]

        self.counter += 1
        return e, batch_x, batch_y



if __name__ == "__main__":
    dataset = Dataset('./data', batch_size=10, sequence_length=10)
    dataset.preprocess(input_file)
    batches_x, batches_y = dataset.create_minibatches()
    print(batches_x.shape)
