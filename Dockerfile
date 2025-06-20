FROM python:3.10-slim
WORKDIR /app
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir spikingjelly
COPY best_model_net_4.pth .
COPY net4.py .
COPY data/ ./data

ENTRYPOINT [ "python", "net4.py"]