# Sinan Utku Ulu
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import pandas as pd
import matplotlib as plt
import numpy as np
import re
from autoencoder import Autoencoder
from critic import Critic
from generator import Generator
from collections import Counter
import tqdm

TRAIN_PATH = 'data_snli/train.txt'
TEST_PATH = 'data_snli/test.txt'

with open(TRAIN_PATH, 'rb') as f:
    text = f.read().lower().decode(encoding='utf-8')

def clean_text(text):
    pattern = r'[^a-zA-Z,.0-9 ]'
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text

text = clean_text(text)

chars = sorted(list(set(text)))  # Get a list of unique characters

# Create dictionaries for character to index mapping and vice versa
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

encoded_text = [char_to_idx[ch] for ch in text]

# Function to create input sequences and their corresponding targets
def create_sequences(encoded_text, seq_length):
    inputs = []
    targets = []

    for i in range(0, len(encoded_text) - seq_length):
        inputs.append(encoded_text[i : i + seq_length])
        targets.append(encoded_text[i + seq_length])

    return inputs, targets

seq_length = 32
inputs, targets = create_sequences(encoded_text, seq_length)


class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]

        return x, y

    def __len__(self):
        return len(self.data)

def greedy_search(conditional_probability):
    return (np.argmax(conditional_probability))

# Assume we have some data in the form of NumPy arrays or PyTorch tensors
data = torch.FloatTensor(inputs)
labels = torch.FloatTensor(targets)

# Create an instance of our custom Dataset
dataset = CustomDataset(data, labels)

# Create a DataLoader
batch_size = 32
data_loader = DataLoader(dataset, batch_size=batch_size,shuffle=True)

# Generator and Discrimnator
n_epochs = 20000
z_dim = 32
display_step = 500
batch_size = 32
lr = 0.0001
beta_1 = 0.5
beta_2 = 0.9
c_lambda = 10
crit_repeats = 5
# Autoencoder model parameters
batch_size_ae = 128
lr_ae = 0.001
beta_1_ae = 0.9
beta_2_ae = 0.9
device = 'cpu'

gen = Generator(z_dim).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(beta_1, beta_2))
crit = Critic().to(device)
crit_opt = torch.optim.Adam(crit.parameters(), lr=lr, betas=(beta_1, beta_2))

ae = Autoencoder().to(device)
ae_opt = torch.optim.Adam(ae.parameters(), lr=lr_ae, betas=(beta_1_ae, beta_2_ae))


def weights_init(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm1d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)


gen = gen.apply(weights_init)
crit = crit.apply(weights_init)
ae = ae.apply(weights_init)


def get_gradient(crit, real, fake, epsilon):
    mixed_text = real * epsilon + fake * (1 - epsilon)

    mixed_scores = crit(mixed_text)

    gradient = torch.autograd.grad(
        inputs=mixed_text,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    return gradient


def gradient_penalty(gradient):
    gradient = gradient.view(len(gradient), -1)

    gradient_norm = gradient.norm(2, dim=1)

    penalty = torch.pow(torch.mean(gradient_norm - 1), 2)

    return penalty

def get_ae_loss(x, output):
    ae_loss = -torch.mean(np.power(np.linalg.norm(x-output), 2))
    return ae_loss

def get_gen_loss(crit_fake_pred):
    gen_loss = -torch.mean(crit_fake_pred)
    return gen_loss

def get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda, ae_loss):
    crit_loss = torch.mean((c_lambda * gp) - torch.mean(crit_real_pred) + torch.mean(crit_fake_pred) + ae_loss )
    return crit_loss
def get_noise(n_samples, z_dim, device='cpu'):
    return torch.randn(n_samples, z_dim, device=device)

k = 5
cur_step = 0
generator_losses = []
critic_losses = []
ae_losses = []
for epoch in range(n_epochs):
    for real, _ in data_loader:
        ae_opt.zero_grad()
        ae_out = ae(real)
        ae_loss = get_ae_loss(real, ae_out)
        ae_loss.backward()
        ae_opt.step()
        ae_losses += [ae_loss.item()]
        cur_batch_size = len(real)
        real = real.to(device)

        mean_iteration_critic_loss = 0
        for _ in range(crit_repeats):
            crit_opt.zero_grad()
            fake_noise = get_noise(cur_batch_size, z_dim, device=device)
            fake = gen(fake_noise)
            crit_fake_pred = crit(fake.detach())
            crit_real_pred = crit(ae_out)

            epsilon = torch.rand(len(real), 1, 1, 1, device=device, requires_grad=True)
            gradient = get_gradient(crit, ae_out, fake.detach(), epsilon)
            gp = gradient_penalty(gradient)
            crit_loss = get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda, ae_loss)

            mean_iteration_critic_loss += crit_loss.item() / crit_repeats

            crit_loss.backward(retain_graph=True)

            crit_opt.step()
        critic_losses += [mean_iteration_critic_loss]

        gen_opt.zero_grad()
        fake_noise_2 = get_noise(cur_batch_size, z_dim, device=device)
        fake_2 = gen(fake_noise_2)
        crit_fake_pred = crit(fake_2)

        gen_loss = get_gen_loss(crit_fake_pred)
        gen_loss.backward()

        gen_opt.step()

        generator_losses += [gen_loss.item()]

        if cur_step % display_step == 0 and cur_step > 0:
            gen_mean = sum(generator_losses[-display_step:]) / display_step
            crit_mean = sum(critic_losses[-display_step:]) / display_step
            print(f"Epoch {epoch}, step {cur_step}: Generator loss: {gen_mean}, critic loss: {crit_mean}")

            step_bins = 20
            num_examples = (len(generator_losses) // step_bins) * step_bins

            plt.plot(
                range(num_examples // step_bins),
                torch.Tensor(ae_losses[:num_examples]).view(-1, step_bins).mean(1),
                label="Autoencoder Loss"
            )
            plt.plot(
                range(num_examples // step_bins),
                torch.Tensor(generator_losses[:num_examples]).view(-1, step_bins).mean(1),
                label="Generator Loss"
            )
            plt.plot(
                range(num_examples // step_bins),
                torch.Tensor(critic_losses[:num_examples]).view(-1, step_bins).mean(1),
                label="Critic Loss"
            )
            plt.legend()
            plt.show()

        cur_step += 1


