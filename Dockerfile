FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV DISPLAY=:0
ENV WINEARCH=win64
ENV WINEPREFIX=/root/.wine

RUN apt-get update && apt-get install -y \
    wget curl python3 python3-pip \
    wine64 wine32 winetricks \
    xvfb \
    libpulse0 \
    && apt-get clean

RUN pip3 install --upgrade pip
RUN pip3 install mt5linux pandas numpy

WORKDIR /app
COPY . .

RUN chmod +x start.sh

CMD ["bash", "start.sh"]
