import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, surrogate, encoding, layer, functional

import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


################################################################################
class Flatten(nn.Module):
  def forward(self, x):
    return x.view(x.shape[0], -1)
tau = 2.0
learning_rate = 1e-3
encoder = encoding.PoissonEncoder()

net_4 = nn.Sequential(
    #(T, 64, 1, 28, 28)
    Flatten(),                         # (B, 1*28*28)
    layer.Linear(28*28, 100, bias=False),
    neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),
    nn.Dropout(0.2),
    layer.Linear(100, 10, bias=False),
    neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan())
    ).to(device)

checkpoint = torch.load('best_model_net_4.pth', map_location=device, weights_only=True)
net_4.load_state_dict(checkpoint['model_state_dict'])
################################################################################

print("Success!!!")

from torchvision import transforms as T

transform = T.Compose([
    T.Grayscale(num_output_channels=1),   # → 1-channel PIL image
    T.Resize((28, 28)),                   # → 28×28 pixels
    T.ToTensor(),                         # → tensor, shape (1,28,28), dtype float32
])
net_4.eval()



from PIL import Image
import argparse, torch
import time
from torch import amp
import torch.nn.functional as F
scaler = amp.GradScaler('cuda')
T = 30

@torch.no_grad()
def predict(path: str):
    img = Image.open(path).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)
    with amp.autocast('cuda'):
        out_fr = 0.
        for t in range(T):
          encoded_img = encoder(img).float()
          out_fr += net_4(encoded_img)
        out_fr = out_fr / T
    #print(out_fr)
    functional.reset_net(net_4)
    return out_fr.argmax(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="path to jpg / png")
    args = parser.parse_args()
    print("Here is the classification for your image:")
    label = predict(args.image)
    print(f"{args.image}  ➜  {label}")