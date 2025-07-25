import pandas as pd
import uuid
from datos import cargar_datos_crudos

# Data Rapida
def generar_descripciones_dataframe(df: pd.DataFrame, nivel: str) -> pd.DataFrame:
    prompts = {
        "municipal": lambda r: f"Describe esta observación: En el municipio de {r['municipio']}, del departamento de {r['departamento']}, en el año {r['año']}, el indicador '{r['indicador']}' tuvo un valor de {r['dato_municipio']} con tipo de medida {r['tipo_de_medida']}.",
        "departamental": lambda r: f"Describe esta observación: En el departamento de {r['departamento']}, en el año {r['año']}, el indicador '{r['indicador']}' tuvo un valor de {r['dato_departamento']} con tipo de medida {r['tipo_de_medida']}.",
        "regional": lambda r: f"Describe esta observación: En la región {r['region']}, en el año {r['año']}, el indicador '{r['indicador']}' tuvo un valor de {r['dato_region']} con tipo de dato {r['tipo_dato']}."
    }

    campo = {
        "municipal": "dato_municipio",
        "departamental": "dato_departamento",
        "regional": "dato_region"
    }[nivel]

    df = df.dropna(subset=[campo, "indicador"]).copy().head(500)  # Puedes ajustar este límite
    df["id"] = [str(uuid.uuid4()) for _ in range(len(df))]
    df["respuesta_llama"] = df.apply(lambda r: prompts[nivel](r), axis=1)
    return df

# Guardar 
def procesar_todo():
    df_mun, df_dep, df_reg = cargar_datos_crudos()

    df_mun_desc = generar_descripciones_dataframe(df_mun, "municipal")
    df_mun_desc.to_excel("data/descripcion_llama_municipales.xlsx", index=False)

    df_dep_desc = generar_descripciones_dataframe(df_dep, "departamental")
    df_dep_desc.to_excel("data/descripcion_llama_departamentales.xlsx", index=False)

    df_reg_desc = generar_descripciones_dataframe(df_reg, "regional")
    df_reg_desc.to_excel("data/descripcion_llama_regionales.xlsx", index=False)

    print("Descripciones generadas y guardadas sin modelo.")

if __name__ == "__main__":
    procesar_todo()
