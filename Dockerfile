# app/Dockerfile

# Utiliser l'image Python 3.9 slim comme base
FROM python:3.9-slim

# Définir le répertoire de travail à /app
WORKDIR /app

# Installation des paquets nécessaires
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Cloner le dépôt git dans le répertoire de travail actuel
RUN git clone https://github.com/Alex-s4d/scoring_credit.git .
RUN git pull

# Installer les dépendances Python
RUN pip install plotly
RUN pip install cufflinks
RUN pip3 install -r requirements.txt
RUN pip3 cache purge


# Ajouter le répertoire local de l'utilisateur à PATH pour inclure les exécutables
ENV PATH="${PATH}:/root/.local/bin"

# Exposer le port 8501 pour le tableau de bord Streamlit
EXPOSE 8501

# Exposer le port 8000 pour l'API
EXPOSE 8000

# Vérifier l'état de l'application avec curl (pour la santé de l'API)
HEALTHCHECK CMD curl --fail http://localhost:8000/_health || exit 1

# Commande d'entrée pour lancer à la fois Streamlit et l'API
CMD streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0 & python3 -m api.app --port 8000
