# we start from python image
FROM python:3.10-slim

# create app directory 
WORKDIR /app

# install dependencies
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir spikingjelly

# copy needed files
COPY train.py .
COPY train_data ./train_data

# set the entry point
ENTRYPOINT [ "python", "train.py"]