import streamlit as st
import pandas as pd
from api import predict
import plotly.graph_objects as go
import json
import ast

def main():
    st.title('Dashboard Scoring crédit :')

    id_file = "data/preprocessing_test.csv"
    df = pd.read_csv(id_file, nrows=2000)

    # Choix entre drag and drop et sélection d'un ID
    choix = st.radio("Choisir une option :", ("Drag and drop", "Sélectionner un ID"))

    if choix == "Drag and drop":
        # Sélectionner le fichier CSV à utiliser pour les prédictions
        uploaded_file = st.file_uploader("Choisir un fichier CSV", type=['csv'])

        if uploaded_file is not None:
            st.write('Fichier chargé avec succès.')
            
            # Afficher les 5 premières lignes du fichier CSV chargé
            ids_df = pd.read_csv(uploaded_file)

            # Bouton pour effectuer les prédictions
        if st.button('Effectuer les prédictions'):
            with st.spinner('Prédiction en cours...'):
                # Exécuter la fonction de prédiction avec le fichier CSV chargé
                prediction = predict.main(ids_df)

            # Afficher la prédiction
            st.subheader('Prédiction :')
            st.write(prediction['prediction'])

            # Afficher les caractéristiques importantes
            feature_importances = prediction['feature_importances']

            if feature_importances is not None:
                # Extraire les noms des fonctionnalités et leurs importances
                feature_names = [feature['Feature'] for feature in feature_importances]
                importances = [feature['Importance'] for feature in feature_importances]

                # Sélectionnez les 10 caractéristiques les plus importantes
                top_features = feature_names[:10]
                top_importances = importances[:10]

                # Créez un graphique à barres pour afficher les caractéristiques importantes
                fig = go.Figure(data=[go.Bar(x=top_features, y=top_importances)])
                fig.update_layout(title='Les 10 caractéristiques les plus importantes du modèle',
                                xaxis_title='Caractéristiques',
                                yaxis_title='Importance')

                # Affichez le graphique dans Streamlit
                st.plotly_chart(fig)
            else:
                st.write("Aucune donnée sur l'importance des fonctionnalités disponible.")

            # Extract the names of the top 10 features
            top_feature_names = [feature['Feature'] for feature in feature_importances][:10]

            # Initialize a list to store the feature names and values
            feature_values_list = []

            # Search for the values of the top 10 features in prediction['feature_values']
            for feature in top_feature_names:
                for item in prediction['feature_values']:
                    if item['Feature'] == feature:
                        feature_values_list.append({'Feature': feature, 'Value': item['Values']})

            feature_values_df = pd.DataFrame(feature_values_list)

            feature_values_mean = df[top_feature_names].mean()

            # Calculate the mean of feature values for 'TARGET' == 0
            feature_values_mean_target_0 = df[df['TARGET'] == 0][top_feature_names].mean()

            # Calculate the mean of feature values for 'TARGET' == 1
            feature_values_mean_target_1 = df[df['TARGET'] == 1][top_feature_names].mean()
            # Add these means to the feature_values_df DataFrame
            feature_values_df['Mean'] = feature_values_mean.values
            feature_values_df['Mean_Target_0'] = feature_values_mean_target_0.values
            feature_values_df['Mean_Target_1'] = feature_values_mean_target_1.values

            for feature_name in top_feature_names:
                st.subheader(f"Statistiques pour '{feature_name}'")

                # Get the feature value and its mean
                feature_value = feature_values_df.loc[feature_values_df['Feature'] == feature_name, 'Value'].iloc[0]
                feature_mean = df[feature_name].mean()
                feature_mean_target_0 = df.loc[df['TARGET'] == 0, feature_name].mean()
                feature_mean_target_1 = df.loc[df['TARGET'] == 1, feature_name].mean()

                # Create a bar chart for the feature value
                x = ['Client', 'Moyenne global', "Moyenne des clients sans défaut", "Moyenne des clients en défaut"]
                y = [feature_value, feature_mean, feature_mean_target_0, feature_mean_target_1]

                # Définition des couleurs pour chaque barre
                couleurs = ['blue', 'green', 'orange', 'red']

                # Création de la figure
                fig = go.Figure()

                # Ajout de chaque trace avec une couleur spécifique
                for i in range(len(x)):
                    fig.add_trace(go.Bar(
                        x=[x[i]],
                        y=[y[i]],
                        name=x[i],  # Utilisation du nom comme légende
                        marker_color=couleurs[i]  # Utilisation de la couleur spécifique
                    ))

                # Ajout d'une légende
                fig.update_layout(
                    legend_title_text='Statistiques',  # Titre de la légende
                )

                # Display the chart in Streamlit
                st.plotly_chart(fig)

                # Display the chart in Streamlit
                st.plotly_chart(fig)

    else:
        # Charger le fichier CSV interne au projet contenant les IDs

        # Afficher les IDs disponibles
        selected_id = st.selectbox("Sélectionner un ID :", df['SK_ID_CURR'].tolist())

        # Afficher l'ID sélectionné
        st.write("ID sélectionné :", selected_id)

        ids_df = df[df['SK_ID_CURR'] == selected_id]

                # Bouton pour effectuer les prédictions
        if st.button('Effectuer les prédictions'):
            with st.spinner('Prédiction en cours...'):
                # Exécuter la fonction de prédiction avec le fichier CSV chargé
                prediction = predict.main(ids_df)

            # Afficher la prédiction
            st.subheader('Prédiction :')
            st.write(prediction['prediction'])

            # Afficher les caractéristiques importantes
            feature_importances = prediction['feature_importances']

            if feature_importances is not None:
                # Extraire les noms des fonctionnalités et leurs importances
                feature_names = [feature['Feature'] for feature in feature_importances]
                importances = [feature['Importance'] for feature in feature_importances]

                # Sélectionnez les 10 caractéristiques les plus importantes
                top_features = feature_names[:10]
                top_importances = importances[:10]

                # Créez un graphique à barres pour afficher les caractéristiques importantes
                fig = go.Figure(data=[go.Bar(x=top_features, y=top_importances)])
                fig.update_layout(title='Les 10 caractéristiques les plus importantes du modèle',
                                xaxis_title='Caractéristiques',
                                yaxis_title='Importance')

                # Affichez le graphique dans Streamlit
                st.plotly_chart(fig)
            else:
                st.write("Aucune donnée sur l'importance des fonctionnalités disponible.")

            # Extract the names of the top 10 features
            top_feature_names = [feature['Feature'] for feature in feature_importances][:10]

            # Initialize a list to store the feature names and values
            feature_values_list = []

            # Search for the values of the top 10 features in prediction['feature_values']
            for feature in top_feature_names:
                for item in prediction['feature_values']:
                    if item['Feature'] == feature:
                        feature_values_list.append({'Feature': feature, 'Value': item['Values']})

            feature_values_df = pd.DataFrame(feature_values_list)

            feature_values_mean = df[top_feature_names].mean()

            # Calculate the mean of feature values for 'TARGET' == 0
            feature_values_mean_target_0 = df[df['TARGET'] == 0][top_feature_names].mean()

            # Calculate the mean of feature values for 'TARGET' == 1
            feature_values_mean_target_1 = df[df['TARGET'] == 1][top_feature_names].mean()
            # Add these means to the feature_values_df DataFrame
            feature_values_df['Mean'] = feature_values_mean.values
            feature_values_df['Mean_Target_0'] = feature_values_mean_target_0.values
            feature_values_df['Mean_Target_1'] = feature_values_mean_target_1.values

            for feature_name in top_feature_names:
                st.subheader(f"Statistiques pour '{feature_name}'")

                # Get the feature value and its mean
                feature_value = feature_values_df.loc[feature_values_df['Feature'] == feature_name, 'Value'].iloc[0]
                feature_mean = df[feature_name].mean()
                feature_mean_target_0 = df.loc[df['TARGET'] == 0, feature_name].mean()
                feature_mean_target_1 = df.loc[df['TARGET'] == 1, feature_name].mean()

                # Create a bar chart for the feature value
                x = ['Client', 'Moyenne global', "Moyenne des clients sans défaut", "Moyenne des clients en défaut"]
                y = [feature_value, feature_mean, feature_mean_target_0, feature_mean_target_1]

                # Définition des couleurs pour chaque barre
                couleurs = ['blue', 'green', 'orange', 'red']

                # Création de la figure
                fig = go.Figure()

                # Ajout de chaque trace avec une couleur spécifique
                for i in range(len(x)):
                    fig.add_trace(go.Bar(
                        x=[x[i]],
                        y=[y[i]],
                        name=x[i],  # Utilisation du nom comme légende
                        marker_color=couleurs[i]  # Utilisation de la couleur spécifique
                    ))

                # Ajout d'une légende
                fig.update_layout(
                    legend_title_text='Statistiques',  # Titre de la légende
                )

                # Display the chart in Streamlit
                st.plotly_chart(fig)






if __name__ == '__main__':
    main()
