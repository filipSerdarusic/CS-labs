from dataset_demo import *
from RNN import *


BATCH_SIZE = 10
SEQ_LENGTH = 20
input_file = './data/selected_conversations.txt'

dataset = Dataset('./data', batch_size=BATCH_SIZE, sequence_length=SEQ_LENGTH)

dataset.preprocess(input_file)
dataset.create_minibatches()

run_language_model(dataset, max_epochs=10, hidden_size=100)