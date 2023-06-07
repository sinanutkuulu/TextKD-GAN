import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

TRAIN_PATH = 'data_snli/train.txt'
TEST_PATH = 'data_snli/test.txt'

with open(TRAIN_PATH, 'rb') as f:
    text = f.read().decode(encoding='utf-8')

chars = sorted(list(set(text)))  # Get a list of unique characters (85)

# Create dictionaries for character to index mapping and vice versa
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

encoded_text = [char_to_idx[ch] for ch in text]

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        # Returns a single instance of your data at the given index
        x = self.data[index]
        y = self.labels[index]

        return x, y

    def __len__(self):
        # Returns the size of the dataset
        return len(self.data)

# Function to create input sequences and their corresponding targets
def create_sequences(encoded_text, seq_length):
    inputs = []
    targets = []

    for i in range(0, len(encoded_text) - seq_length):
        inputs.append(encoded_text[i : i + seq_length])
        targets.append(encoded_text[i + seq_length])

    return inputs, targets

def greedy_search(conditional_probability):
    return (np.argmax(conditional_probability))

seq_length = 32  # Length of input sequence
inputs, targets = create_sequences(encoded_text, seq_length)

# Assume we have some data in the form of NumPy arrays or PyTorch tensors
data = ...
labels = ...

# Create an instance of our custom Dataset
dataset = CustomDataset(data, labels)

# Create a DataLoader
batch_size = 32
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

