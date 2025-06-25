# net4
Training MNIST classifier

How to use it: python train.py batch_size

How to create docker image and run it:

Building: docker build -t training_network:latest .

Running: docker run --gpus all training_network:latest batch_size

Note: The accuracy and the training time depends on the batch size since we are using SNN network.

