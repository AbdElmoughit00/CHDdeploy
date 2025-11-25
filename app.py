# Application Streamlit pour la prédiction du risque cardiaque
import streamlit as st
import pandas as pd
import joblib
import numpy as np
def nettoyer_texte(x):
    # نفس الكود الذي كان مستخدم عند entraînement
    return x.lower().strip()
# Fonction de normalisation textuelle
def normalize_text(X):
    """Normalise le texte de la colonne famhist"""
    X_copy = X.copy()
    if 'famhist' in X_copy.columns:
        X_copy['famhist'] = X_copy['famhist'].str.lower().str.strip()
    return X_copy

# Alias pour compatibilité
clean_text = normalize_text

# Chargement du modèle
try:
    model = joblib.load(r"Model.pkl")
    st.sidebar.success("Modele charge avec succes")
except Exception as e:
    st.error(f"Erreur lors du chargement: {e}")
    st.stop()

st.title("Prediction du risque cardiaque (CHD)")
st.sidebar.header("Caracteristiques du patient")

def saisir_donnees():
    sbp = st.sidebar.number_input('Pression Systolique (sbp)', 100, 220, 130)
    ldl = st.sidebar.number_input('Cholesterol LDL', 0.0, 15.0, 4.0)
    adiposity = st.sidebar.number_input('Adiposite', 0.0, 50.0, 25.0)
    obesity = st.sidebar.number_input('Obesite (BMI)', 10.0, 50.0, 25.0)
    age = st.sidebar.number_input('Age', 15, 80, 45)
    famhist = st.sidebar.selectbox('Antecedent Familial', ['Present', 'Absent'])

    donnees_patient = {
        'sbp': sbp,
        'ldl': ldl,
        'adiposity': adiposity,
        'famhist': famhist,
        'obesity': obesity,
        'age': age
    }
    return pd.DataFrame(donnees_patient, index=[0])

df_saisie = saisir_donnees()

st.subheader('Recapitulatif des donnees')
st.write(df_saisie)

if st.button('Lancer la prediction'):
    try:
        resultat = model.predict(df_saisie)
        probabilites = model.predict_proba(df_saisie)

        if resultat[0] == 1:
            st.error(f"Risque ELEVE de maladie cardiaque (Probabilite: {probabilites[0][1]:.1%})")
        else:
            st.success(f"Risque FAIBLE de maladie cardiaque (Probabilite: {probabilites[0][0]:.1%})")
    except Exception as e:
        st.error(f"Erreur lors de la prediction: {e}")
