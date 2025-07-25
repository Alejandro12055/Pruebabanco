import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel

# Cargar modelo 
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()

def generar_embeddings(nombre: str, ruta_excel: str):
    df = pd.read_excel(ruta_excel)
    if "respuesta_llama" not in df.columns:
        print(f"[ERROR] No se encontró la columna 'respuesta_llama' en {ruta_excel}")
        return
    
    df["descripcion"] = df["respuesta_llama"]
    
    embeddings = []
    for texto in df["descripcion"]:
        inputs = tokenizer(texto, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            emb = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        embeddings.append(emb)
    
    df["embedding"] = [e.tolist() for e in embeddings]
    df.to_json(f"data/descripciones_con_embeddings_{nombre}.json", orient="records", indent=2)
    np.save(f"data/embeddings_{nombre}.npy", np.array(embeddings))
    print(f" Embeddings guardados para {nombre}")

if __name__ == "__main__":
    generar_embeddings("municipales", "data/descripcion_llama_municipales.xlsx")
    generar_embeddings("departamentales", "data/descripcion_llama_departamentales.xlsx")
    generar_embeddings("regionales", "data/descripcion_llama_regionales.xlsx")
