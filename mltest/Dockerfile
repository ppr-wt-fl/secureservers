# compare.py's client and oracle side: torch (CPU) and pandas for fedavg.py, cryptography for shares.py.
# The code is mounted, not copied (compose.mltest.yaml).
FROM python:3.13-slim
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
        -r /tmp/requirements.txt cryptography==50.0.2
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
