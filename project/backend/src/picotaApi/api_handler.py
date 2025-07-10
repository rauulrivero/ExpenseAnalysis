import requests
import json
import numpy as np
import pandas as pd
from time import sleep

def inference(payloads, ccaa, api_id, delay=0):
    """
    Envía uno o varios payloads (diccionarios) a la API y devuelve un DataFrame con las predicciones.

    Parameters
    ----------
    payloads : dict o list of dict
        Datos del hogar o lista de hogares para enviar a la API.
    ccaa : str
        Código de comunidad autónoma (por ejemplo "01").
    api_id : str
        ID de la API en picota.io.
    delay : int, optional
        Tiempo de espera entre peticiones.

    Returns
    -------
    pd.DataFrame
        DataFrame con predicciones por hogar.
    """
    if isinstance(payloads, dict):
        payloads = [payloads]  # Convertir a lista si es un único hogar

    url = f"https://picota.io/api/1.0.0/digital-twin/{api_id}/subject/hogar{ccaa}/inference"
    predicciones = []

    bools_cols = ['capitalProvincia', 'aguaCaliente', 'calefaccion', 'espanolSp', 'educacionSuperiorSp']

    for i, payload in enumerate(payloads):
        # Normalizar booleanos
        for col in bools_cols:
            if col in payload:
                valor = payload[col]
                payload[col] = (str(valor).lower() == "true") if isinstance(valor, str) else bool(valor)

        # Asignar campo temporal
        payload["instant"] = payload.pop("timestamp", f"hogar{ccaa}-row{i}")

        try:
            response = requests.post(url, headers={"Content-Type": "application/json"}, data=json.dumps(payload))
            response.raise_for_status()
            data = response.json()
            if data:
                df_pred = pd.DataFrame(data)
                df_pred["hogar"] = i
                predicciones.append(df_pred)
        except Exception as e:
            print(f"⚠️ Error en hogar {i} (CCAA {ccaa}): {e}")

        if delay:
            sleep(delay)

    if not predicciones:
        return pd.DataFrame()

    df_total = pd.concat(predicciones, ignore_index=True)
    df_wide = df_total.pivot_table(index="hogar", columns="variable", values="value")
    df_wide = df_wide.reindex(range(len(payloads)), fill_value=np.nan)
    df_wide["subject"] = f"hogar{ccaa}"
    return df_wide
