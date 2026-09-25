import json
import sys
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.segmentacao import definir_segmento

app = FastAPI()

with open(Path(__file__).resolve().parent / "politica_ts.json") as f:
    politica = json.load(f)


class ClienteRequest(BaseModel):
    age: int
    job: str
    education: str
    previous: int


@app.post("/recomendar")
def recomendar(cliente: ClienteRequest):
    segmento = definir_segmento(cliente.age)
    bracos_segmento = politica[segmento]

    # amostra da Beta em tempo real — é o que torna isso um bandit, não uma tabela congelada
    amostras = {
        braco: np.random.beta(dados["alpha"], dados["beta"])
        for braco, dados in bracos_segmento.items()
    }
    braco_escolhido = max(amostras, key=amostras.get)
    dados_escolhido = bracos_segmento[braco_escolhido]
    canal, dia = braco_escolhido.rsplit("_", 1)
    prob_estimada = dados_escolhido["alpha"] / (dados_escolhido["alpha"] + dados_escolhido["beta"])

    return {
        "segmento": segmento,
        "braco_recomendado": braco_escolhido,
        "canal": canal,
        "dia": dia,
        "prob_estimada": round(prob_estimada, 3),
        "alpha": dados_escolhido["alpha"],
        "beta": dados_escolhido["beta"],
        "n_pulls": dados_escolhido["n_pulls"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}