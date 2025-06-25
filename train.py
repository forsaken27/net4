import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from spikingjelly.activation_based import neuron, functional, surrogate, encoding, layer

import torchvision.datasets as dset
import torchvision.transforms as T

import numpy as np
import sys

USE_GPU = False
dtype = torch.float32

batch_size = int(sys.argv[1])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(">>> Using device on:", device)

transform = T.Compose([
                T.ToTensor(),
            ])
train_dataset = dset.MNIST(root='./train_data', train=True, download=True, transform=transform)
test_dataset = dset.MNIST(root='./train_data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("***************************************************")
print("Here is the shapes of the train and test datasets: ")
print(train_dataset.data.shape)
print(test_dataset.data.shape)
print("***************************************************")











class Flatten(nn.Module):
  
  def forward(self, x):
    return x.view(x.shape[0], -1)
tau = 2.0
learning_rate = 1e-3


net_4 = nn.Sequential(
    #(T, 64, 1, 28, 28)
    Flatten(),                         # (B, 1*28*28)
    layer.Linear(28*28, 100, bias=False),
    neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),
    nn.Dropout(0.2),
    layer.Linear(100, 10, bias=False),
    neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan())
    ).to(device)



optimizer = optim.Adam(net_4.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
encoder = encoding.PoissonEncoder()
epochs = 10










import time
from torch import amp
import torch.nn.functional as F
scaler = amp.GradScaler('cuda')
T = 30

def train_net(T, epochs):
  print(">>> Model on:", next(net_4.parameters()).device)
  net_4.train()
  for epoch in range(epochs):
    start_time = time.time()
    net_4.train()
    train_loss = 0
    train_acc = 0
    train_samples = 0
    for img, label in train_loader:
      optimizer.zero_grad()
      img = img.to(device)
      label = label.to(device)
      label_onehot = F.one_hot(label, 10).float()

          # Mixed-precision training
      if scaler is not None:
        with amp.autocast('cuda'):
          out_fr = 0.
          # Run T time steps
          for t in range(T):
            img = img.to(device)
            encoded_img = encoder(img)
            out_fr += net_4(encoded_img).to(device)
          out_fr = out_fr / T
          # out_fr is tensor whose shape is [batch_size, 10]
          # The firing rate of 10 neurons in the output layer was recorded during the whole simulation period
          loss = F.mse_loss(out_fr, label_onehot)
          # The loss function is the MSE between the firing rate of the output layer and the true category.
        scaler.scale(loss).backward(retain_graph=True)
        scaler.step(optimizer)
        scaler.update()
      else:
        out_fr = 0.
        for t in range(T):
          encoded_img = encoder(img)
          out_fr += net_4(encoded_img)
        out_fr = out_fr / T
        loss = F.mse_loss(out_fr, label_onehot)
        loss.backward()
        optimizer.step()

      train_samples += label.numel()

      train_loss += loss.item() * label.numel()
      # The correct rate is calculated as follows. The subscript i of the neuron with the highest firing rate in the output layer is considered as the result of classification.
      train_acc += (out_fr.argmax(1) == label).float().sum().item()

      # After optimizing the parameters, the state of the network should be reset because the neurons of the SNN have “memory”.
      functional.reset_net(net_4)
    print(f"Epoch: {epoch}")
    print(f"Train Loss: {train_loss / train_samples}")
    print(f"Train Acc: {train_acc /  train_samples}")
    print(f"Time: {time.time() - start_time}")

print()
print("____________________we are starting the training_____________________")
train_net(T, epochs)
print("____________________we finished the training_____________________")










def test_net():
  net_4.eval()                         # evaluation mode
  test_acc = test_samples = 0
  T = 20
  start_time = time.time()

  with torch.no_grad():
      for img, label in test_loader:
          img,  label  = img.to(device), label.to(device)


          # ----- forward -----
          with amp.autocast('cuda'):
              out_fr = 0.
              for t in range(T):
                  encoded_img = encoder(img)
                  out_fr += net_4(encoded_img)
              out_fr = out_fr / T              # average firing rate

          # ----- metrics -----
          test_samples += label.size(0)
          test_acc     += (out_fr.argmax(1) == label).sum().item()

          functional.reset_net(net_4)          # clear membranes for next batch

  print(f"Test Acc: {test_acc / test_samples:.4f}")
  print(f"Time: {time.time() - start_time:.2f}s")

print()
print("____________________we are starting the test_____________________")
test_net()
print("____________________we finished the test_____________________")







