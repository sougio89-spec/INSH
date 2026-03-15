FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV DISPLAY=:0
ENV WINEARCH=win64
ENV WINEPREFIX=/root/.wine

# Installation Wine + dépendances
RUN dpkg --add-architecture i386 && \
    apt-get update && apt-get install -y \
    wget curl python3 python3-pip xvfb libpulse0 \
    gnupg2 software-properties-common cabextract \
    && wget -nc https://dl.winehq.org/wine-builds/winehq.key \
    && apt-key add winehq.key \
    && add-apt-repository 'deb https://dl.winehq.org/wine-builds/ubuntu/ jammy main' \
    && apt-get update \
    && apt-get install -y --install-recommends winehq-stable \
    && apt-get clean

# Installation packages Linux
RUN pip3 install --upgrade pip && \
    pip3 install mt5linux pandas numpy

# Télécharger Python 3.9.10 Windows + installer dans Wine
RUN Xvfb :0 -screen 0 1024x768x16 & sleep 3 && \
    DISPLAY=:0 wineboot --init && sleep 5 && \
    wget -q https://www.python.org/ftp/python/3.9.10/python-3.9.10-amd64.exe -O /tmp/python.exe && \
    DISPLAY=:0 wine /tmp/python.exe /quiet InstallAllUsers=1 PrependPath=1 && \
    sleep 15 && \
    DISPLAY=:0 wine python -m pip install MetaTrader5 && \
    sleep 5

WORKDIR /app
COPY . .
RUN chmod +x start.sh

CMD ["bash", "start.sh"]
