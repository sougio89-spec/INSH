FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV DISPLAY=:0
ENV WINEARCH=win64
ENV WINEPREFIX=/root/.wine

# Installation dépendances + Wine
RUN dpkg --add-architecture i386 && \
    apt-get update && apt-get install -y \
    wget curl python3 python3-pip \
    xvfb libpulse0 \
    gnupg2 software-properties-common \
    && wget -nc https://dl.winehq.org/wine-builds/winehq.key \
    && apt-key add winehq.key \
    && add-apt-repository 'deb https://dl.winehq.org/wine-builds/ubuntu/ jammy main' \
    && apt-get update \
    && apt-get install -y --install-recommends winehq-stable \
    && apt-get clean

# Installation Python packages Linux
RUN pip3 install --upgrade pip && \
    pip3 install mt5linux pandas numpy

# Installation pip dans Wine + MetaTrader5 pour Wine
RUN Xvfb :0 -screen 0 1024x768x16 & sleep 3 && \
    DISPLAY=:0 wineboot --init && sleep 5 && \
    DISPLAY=:0 wine python -m pip install MetaTrader5 || true

WORKDIR /app
COPY . .
RUN chmod +x start.sh

CMD ["bash", "start.sh"]
