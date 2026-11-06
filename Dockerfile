# MP-SPDZ v0.4.4 built from source as upstream's Dockerfile does (machine=semi2k-party.x), plus S0 and S1.
# --build-arg insecure=1: the -DINSECURE build with Fake-Offline.x, for online-only timings (compose.online.yaml)
FROM python:3.13-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
        automake build-essential clang cmake git libboost-dev libboost-thread-dev libgmp-dev libntl-dev \
        libsodium-dev libssl-dev libtool \
    && rm -rf /var/lib/apt/lists/*

ENV MPSPDZ=/usr/src/MP-SPDZ
RUN git clone --depth 1 --branch v0.4.4 --recurse-submodules --shallow-submodules \
        https://github.com/data61/MP-SPDZ $MPSPDZ
WORKDIR $MPSPDZ

ARG insecure=0
RUN echo "CXX = clang++" >> CONFIG.mine \
    && echo "MY_CFLAGS += -I/usr/local/include" >> CONFIG.mine \
    && echo "MY_LDLIBS += -Wl,-rpath -Wl,/usr/local/lib -L/usr/local/lib" >> CONFIG.mine \
    && if [ "$insecure" = 1 ]; then echo "MY_CFLAGS += -DINSECURE" >> CONFIG.mine; fi \
    && make boost libote \
    && make -j"$(nproc)" semi2k-party.x \
    && if [ "$insecure" = 1 ]; then make -j"$(nproc)" Fake-Offline.x; fi \
    && mkdir -p Player-Data Persistence

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY shares.py party.py s0.py s1.py epprfl.mpc /app/
WORKDIR /app
