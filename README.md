# Projet de Scoring Client Bancaire

Ce projet vise à développer un système de scoring client pour une institution bancaire. Le système évaluera le risque associé à l'octroi de prêts en fonction des informations fournies par les clients lien vers le kaghgle : https://www.kaggle.com/c/home-credit-default-risk .

## Objectif

L'objectif principal de ce projet est de créer un modèle de scoring prédictif qui aidera l'institution bancaire à prendre des décisions éclairées concernant l'approbation ou le rejet de demandes de prêts. Le modèle sera basé sur des données historiques de clients, telles que leur historique de crédit, leur historique d'emploi, leurs revenus, etc.

## Structure du Projet

Le projet est structuré de la manière suivante :

1. **input/** : Ce répertoire contient les données brut issu du projet kaggle disponnible à cette adresse : https://www.kaggle.com/c/home-credit-default-risk/data.

2. **data/** : Ce répertoire contient les données nécessaires pour l'entraînement et le test du modèle après le preprocessing réalisé.

3. **notebooks/** : Ce répertoire contient les notebooks Jupyter utilisés pour l'exploration des données, l'entraînement du modèle, l'évaluation des performances, etc.

4. **src/** : Ce répertoire contient le code source du projet, y compris les scripts Python pour le prétraitement des données, l'entraînement du modèle, l'évaluation, etc.

5. **models/** : Ce répertoire contient les modèles entraînés qui seront utilisés pour le scoring client.

6. **README.md** : Ce fichier contient des informations sur le projet, son objectif, sa structure, et des instructions pour son utilisation.


7. **requirements.txt** : Ce fichier contient toutes les dépendences des libairies nécessaires au fonctionnement du projet

8. **output** : Ce répertoire contient le fichier csv de submission de concour kaggle

9. **main.py** : Ce fichier permet de lancer le projet dans son intégralité


## Installation

1. Cloner le dépôt GitHub :

git clone https://github.com/Alex-s4d/Openclassrooms.git

2. Installer les dépendances :

pip install -r requirements.txt


## Utilisation

1. Explorer les notebooks dans le répertoire `notebooks/` pour comprendre le processus de développement du modèle.

2. Exécuter les scripts Python dans le répertoire `src/` pour prétraiter les données, entraîner le modèle, évaluer les performances, etc.

3. Utiliser les modèles entraînés dans le répertoire `models/` pour le scoring des clients.

## Auteurs

- Alexandre Masson







