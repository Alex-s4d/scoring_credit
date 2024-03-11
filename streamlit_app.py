import streamlit as st
import pandas as pd
from api import predict

def main():
    st.title('Dashboard de Prédictions')

    # Sélectionner le fichier CSV à utiliser pour les prédictions
    uploaded_file = st.file_uploader("Choisir un fichier CSV", type=['csv'])

    if uploaded_file is not None:
        st.write('Fichier chargé avec succès.')

        # Afficher les 5 premières lignes du fichier CSV chargé
        df = pd.read_csv(uploaded_file)
        st.write('Aperçu des données :')
        st.write(df.head())

        # Bouton pour effectuer les prédictions
        if st.button('Effectuer les prédictions'):
            with st.spinner('Prédiction en cours...'):
                # Exécuter la fonction de prédiction avec le fichier CSV chargé
                predictions = predict.main(df)

            # Afficher les prédictions
            st.write('Prédictions :')
            st.write(predictions)

if __name__ == '__main__':
    main()