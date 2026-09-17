"""
Script de inferência do modelo de churn (Telco Customer Churn).
Uso: python predict.py caminho_para_novos_clientes.csv
"""
import sys
import json
import pickle
import pandas as pd

with open("artefatos_modelo/pipeline_churn.pkl", "rb") as f:
    pipeline_final = pickle.load(f)
with open("artefatos_modelo/metadados.json", "r", encoding="utf-8") as f:
    metadados = json.load(f)

THRESHOLD = metadados["threshold_decisao"]


def preparar_features(df_bruto):
    d = df_bruto.copy()
    d["TotalCharges"] = pd.to_numeric(d["TotalCharges"], errors="coerce").fillna(0.0)
    if "customerID" in d.columns:
        d = d.drop(columns=["customerID"])
    if "Churn" in d.columns:
        d = d.drop(columns=["Churn"])
    servicos_adicionais = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                            "TechSupport", "StreamingTV", "StreamingMovies"]
    d["num_servicos_adicionais"] = (d[servicos_adicionais] == "Yes").sum(axis=1)
    d["cobranca_media_mensal_historica"] = d["TotalCharges"] / d["tenure"].replace(0, 1)
    for col in servicos_adicionais:
        d[col] = d[col].replace("No internet service", "No")
    d["MultipleLines"] = d["MultipleLines"].replace("No phone service", "No")
    for col in ["Partner", "Dependents", "PhoneService", "PaperlessBilling", "MultipleLines"] + servicos_adicionais:
        d[col] = (d[col] == "Yes").astype(int)
    return d


def prever(caminho_csv):
    df_novos = pd.read_csv(caminho_csv)
    ids = df_novos["customerID"] if "customerID" in df_novos.columns else df_novos.index
    X_novos = preparar_features(df_novos)
    probabilidades = pipeline_final.predict_proba(X_novos)[:, 1]
    previsoes = (probabilidades >= THRESHOLD).astype(int)
    resultado = pd.DataFrame({
        "customerID": ids,
        "probabilidade_churn": probabilidades.round(4),
        "previsao_churn": previsoes,
    })
    return resultado


if __name__ == "__main__":
    caminho = sys.argv[1] if len(sys.argv) > 1 else "novos_clientes.csv"
    resultado = prever(caminho)
    resultado.to_csv("previsoes_churn.csv", index=False)
    print(f"{len(resultado)} previsões salvas em previsoes_churn.csv")
