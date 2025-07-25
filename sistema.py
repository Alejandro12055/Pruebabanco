from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import json
import numpy as np
import torch
import re
from transformers import BertTokenizer, BertModel
from langchain.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, pipeline
from sklearn.metrics.pairwise import cosine_similarity
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Pregunta(BaseModel):
    pregunta: str
    nivel: str

# Modelos
modelo_qwen = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen1.5-7B-Chat",
    device_map="cpu",
    torch_dtype=torch.float32,
    trust_remote_code=True
)
tokenizer_qwen = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-7B-Chat", trust_remote_code=True)
streamer = TextStreamer(tokenizer_qwen, skip_prompt=True, skip_special_tokens=True)
pipe = pipeline("text-generation", model=modelo_qwen, tokenizer=tokenizer_qwen, max_new_tokens=512, streamer=streamer)

llm = HuggingFacePipeline(pipeline=pipe)
prompt_template = PromptTemplate.from_template("""
You are a professional on health. You must only answer the question based on the context.

Context:
{contexto}

Reply to this question:
{pregunta}
""")

chain = LLMChain(llm=llm, prompt=prompt_template)

# Embeddings con BERT
bert_model = BertModel.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")
bert_tokenizer = BertTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")

def vectorizar_bert(texto):
    inputs = bert_tokenizer(texto, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def buscar_contexto(embedding_pregunta, embeddings, descripciones, k=5):
    similitudes = cosine_similarity([embedding_pregunta], embeddings)[0]
    indices = np.argsort(similitudes)[-k:][::-1]
    return [descripciones[i]["descripcion"] for i in indices]

# Datos por nivel
DATOS = {
    "municipal": {
        "json": "data/descripciones_con_embeddings_municipales.json",
        "npy": "data/embeddings_municipales.npy",
        "xlsx": "data/descripcion_llama_municipales.xlsx",
    },
    "departamental": {
        "json": "data/descripciones_con_embeddings_departamentales.json",
        "npy": "data/embeddings_departamentales.npy",
        "xlsx": "data/descripcion_llama_departamentales.xlsx",
    },
    "regional": {
        "json": "data/descripciones_con_embeddings_regionales.json",
        "npy": "data/embeddings_regionales.npy",
        "xlsx": "data/descripcion_llama_regionales.xlsx",
    }
}

@app.post("/preguntar")
def responder(input: Pregunta):
    datos = DATOS.get(input.nivel)
    if not datos:
        return {"error": "Nivel no reconocido"}

    # Extraer año y departamento si están en la pregunta
    año_match = re.search(r"(20\d{2})", input.pregunta)
    año = int(año_match.group(1)) if año_match else None
    departamento_match = re.search(r"departamento de ([\wáéíóúñÑ]+)", input.pregunta, re.IGNORECASE)
    departamento = departamento_match.group(1).lower() if departamento_match else None

    with open(datos["json"], "r", encoding="utf-8") as f:
        descripciones = json.load(f)
    embeddings = np.load(datos["npy"])

    descripciones_filtradas = []
    embeddings_filtrados = []
    for i, d in enumerate(descripciones):
        texto = d["descripcion"].lower()
        cumple_año = str(año) in texto if año else True
        cumple_depto = departamento in texto if departamento else True
        if cumple_año and cumple_depto:
            descripciones_filtradas.append(d)
            embeddings_filtrados.append(embeddings[i])

    if not descripciones_filtradas:
        descripciones_filtradas = descripciones
        embeddings_filtrados = embeddings

    emb_pregunta = vectorizar_bert(input.pregunta)
    contexto = buscar_contexto(emb_pregunta, np.array(embeddings_filtrados), descripciones_filtradas)

    resultado = chain.run({"contexto": "\n".join(contexto), "pregunta": input.pregunta})

    return {
        "respuesta": resultado,
        "contexto_utilizado": contexto
    }

@app.get("/resumen")
def resumen(nivel: str = "municipal"):
    datos = DATOS.get(nivel)
    if not datos:
        return {"error": "Nivel no reconocido"}

    df = pd.read_excel(datos["xlsx"])
    try:
        stats = df.describe(include="all").to_dict()
    except Exception as e:
        return {"error": str(e)}
    return stats

@app.get("/keywords")
def keywords(nivel: str = "municipal"):
    datos = DATOS.get(nivel)
    if not datos:
        return {"error": "Nivel no reconocido"}
    with open(datos["json"], "r", encoding="utf-8") as f:
        descripciones = json.load(f)
    textos = " ".join(d["descripcion"] for d in descripciones)
    palabras = textos.lower().split()
    frecuencia = pd.Series(palabras).value_counts().head(10)
    return {"palabras_clave": list(frecuencia.index)}

@app.get("/estadisticas")
def estadisticas(nivel: str = "municipal"):
    datos = DATOS.get(nivel)
    if not datos:
        return {"error": "Nivel no reconocido"}
    df = pd.read_excel(datos["xlsx"])
    return {
        "n_registros": len(df),
        "indicadores_unicos": df["indicador"].nunique(),
        "años_cubiertos": sorted(df["año"].unique().tolist()),
        "media": df.select_dtypes(include=[np.number]).mean().to_dict(),
        "suma": df.select_dtypes(include=[np.number]).sum().to_dict(),
    }
