# importing dependencies
import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, surrogate, encoding, layer, functional
from torchvision import transforms as T

#set the device to gpu if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


################################################################################
###Creating class Flatten to include it inside nn.Sequential module architecture
class Flatten(nn.Module):
  def forward(self, x):
    return x.view(x.shape[0], -1)



learning_rate = 1e-3
tau = 2.0    #for LIF layer
encoder = encoding.PoissonEncoder() 


# Network architecture
# affine -> LIF -> Dropout -> affine -> LIF 
net_4 = nn.Sequential(
    Flatten(),                                                     #(batch_size=64, 1, 28, 28) -> (64, 28*28)
    layer.Linear(28*28, 100, bias=False),                          # -> (64, 100)
    neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),  # -> (64, 100)
    nn.Dropout(0.2),                                               # -> (64, 100)
    layer.Linear(100, 10, bias=False),                             # -> (64, 10)
    neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan())   # -> (64, 10)
    ).to(device)

# The network was trained in google collab and we have the learned weights in
# best_model_net_4.pth
checkpoint = torch.load('best_model_net_4.pth', map_location=device, weights_only=True)
net_4.load_state_dict(checkpoint['model_state_dict'])
################################################################################

print("Success!!!")

#transform to make the input image 28x28 with one channel
transform = T.Compose([
    T.Grayscale(num_output_channels=1),   # → 1-channel PIL image
    T.Resize((28, 28)),                   # → 28×28 pixels
    T.ToTensor(),                         # → tensor, shape (1,28,28), dtype float32
])
net_4.eval()  #setting the model to evaluation mode 



from PIL import Image
import argparse, torch
from torch import amp
import torch.nn.functional as F

# T for LIF layers
num_time = 30

#classifies the input image
@torch.no_grad()
def predict(path: str):
    img = Image.open(path).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)    #transform the image to (1, 1, 1, 28, 28)
    with amp.autocast('cuda'):
        out_fr = 0.
        for _ in range(num_time):                   #we are encoding the image to (T, 1, 1, 28, 28)
          encoded_img = encoder(img).float()        #and giving encoded_img of size (1, 1, 28, 28) each iteration
          out_fr += net_4(encoded_img)
        out_fr = out_fr / num_time
    functional.reset_net(net_4)
    return out_fr.argmax(1)


#we get input image from args and try to classify it 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="path to jpg / png")
    args = parser.parse_args()
    print("Here is the classification for your image:")
    label = predict(args.image)
    print(f"{args.image}  ➜  {label.item()}")