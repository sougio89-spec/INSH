#!/bin/bash

echo "🚀 Démarrage NOVA IA V7.1..."

# Démarrage écran virtuel
Xvfb :0 -screen 0 1024x768x16 &
sleep 2

# Démarrage serveur mt5linux
python3 -m mt5linux &
sleep 3

echo "✅ Serveur MT5Linux démarré"

# Lancement du bot
python3 nova_bot.py
