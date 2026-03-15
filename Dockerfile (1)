FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV DISPLAY=:0
ENV WINEARCH=win64
ENV WINEPREFIX=/root/.wine

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

RUN pip3 install --upgrade pip && \
    pip3 install mt5linux pandas numpy

WORKDIR /app
COPY . .
RUN chmod +x start.sh

CMD ["bash", "start.sh"]
