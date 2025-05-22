from flask import Blueprint, request, jsonify
import json
from model.model import DigitalTwinModel



api = Blueprint('api', __name__)

@api.route('/', methods=['GET'])
def index():
    return jsonify({"message": "Fiscal Simulation"})


# Cargar todo desde el archivo .pt
checkpoint = torch.load("../../../../model/digital_twin_model.pt", map_location=torch.device("cpu"))
input_size = checkpoint['scaler_X'].mean_.shape[0]

# Reconstruir el modelo y cargar pesos
model = DigitalTwinModel(input_size)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Recuperar scaler y parámetros de salida
scaler_X = checkpoint['scaler_X']
y_means = checkpoint['y_means']
y_stds = checkpoint['y_stds']

# --- Ruta de predicción ---
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        inputs = data.get("inputs")

        if inputs is None or len(inputs) != input_size:
            return jsonify({"error": f"'inputs' must be a list of {input_size} values"}), 400

        # Escalar entrada
        input_array = np.array(inputs).reshape(1, -1)
        input_scaled = scaler_X.transform(input_array)
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

        # Inferencia
        with torch.no_grad():
            y_pred_scaled = model(input_tensor).item()

        # Desnormalizar salida
        y_pred = y_pred_scaled * y_stds + y_means

        return jsonify({"prediction": y_pred})

    except Exception as e:
        return jsonify({"error": str(e)}), 500



