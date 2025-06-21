# we start from python image
FROM python:3.10-slim

# create app directory 
WORKDIR /app

# install dependencies
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir spikingjelly

# copy needed files
COPY best_model_net_4.pth .
COPY net4.py .

# set the entry point
ENTRYPOINT [ "python", "net4.py"]