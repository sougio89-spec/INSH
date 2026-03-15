#!/bin/bash

echo "🚀 Démarrage NOVA IA V7.1..."

# Nettoyer ancien lock
rm -f /tmp/.X0-lock

# Démarrage écran virtuel
Xvfb :0 -screen 0 1024x768x16 &
sleep 3

# Initialiser Wine
DISPLAY=:0 wineboot --init
sleep 5

# Télécharger Python Windows
echo "📥 Téléchargement Python Windows..."
wget -q https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe -O /tmp/python-installer.exe

# Installer Python silencieusement dans Wine
echo "⚙️ Installation Python dans Wine..."
DISPLAY=:0 wine /tmp/python-installer.exe /quiet InstallAllUsers=1 PrependPath=1
sleep 15

# Installer MetaTrader5 dans Wine Python
echo "📥 Installation MetaTrader5..."
DISPLAY=:0 wine python -m pip install MetaTrader5
sleep 5

echo "✅ MetaTrader5 installé dans Wine"

# Démarrage serveur mt5linux
python3 -m mt5linux &
sleep 3

echo "✅ Serveur MT5Linux démarré"

# Lancement du bot
python3 nova_bot.py
