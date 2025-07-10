from flask import Blueprint, request, jsonify, current_app
import pandas as pd
from picotaApi.api_handler import inference 

api = Blueprint('api', __name__)

@api.route('/', methods=['GET'])
def index():
    return jsonify({"message": "Fiscal Simulation"})

@api.route('/inferencia', methods=['POST'])
def inferencia():
    try:
        data = request.get_json()

        if not data or 'payload' not in data or 'ccaa' not in data:
            return jsonify({"error": "Se requiere 'payload' y 'ccaa' en el cuerpo del JSON."}), 400

        payload = data['payload']
        ccaa = data['ccaa']
        delay = data.get('delay', 0)

        # Obtener api_id desde configuración
        api_id = current_app.config.get('API_ID')
        if not api_id:
            return jsonify({"error": "No se ha definido 'API_ID' en la configuración de la app."}), 500

        df_result = inference(payload, ccaa, api_id, delay=delay)

        result_json = df_result.reset_index().to_dict(orient="records")
        return jsonify(result_json), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
