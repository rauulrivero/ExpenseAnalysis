from flask import Blueprint, request, jsonify
import json
from src.model.model import EnsembleDigitalTwin
from src.services.child_tax_deduction import TaxSimulator
import torch
import numpy as np



api = Blueprint('api', __name__)

@api.route('/', methods=['GET'])
def index():
    return jsonify({"message": "Fiscal Simulation"})


# Cargar todo desde el archivo .pt
checkpoint = torch.load("model/digital_twin_model.pt", map_location=torch.device("cpu"), weights_only=False)

# Recuperar scaler y parámetros de salida
scaler_X = checkpoint['scaler_X']
y_means = checkpoint['y_means']
y_stds = checkpoint['y_stds']

input_size  = scaler_X.mean_.shape[0]
output_size = y_means.shape[0]
model = EnsembleDigitalTwin(in_dim=input_size, n_outputs=output_size)


# Reconstruir el modelo y cargar pesos
state = torch.load('model/ensemble_final.pth', map_location=torch.device("cpu"))
model.load_state_dict(state)
model.eval()


output_cols = [
    11, 12, 21, 22, 31, 32, 41, 42, 43, 44,
    45, 51, 52, 53, 54, 55, 56, 61, 62, 63,
    71, 72, 73, 81, 82, 83, 91, 92, 93, 94,
    95, 96, 101, 102, 103, 104, 111, 112,
    121, 123, 124, 125, 126, 127, 128
]

feature_cols = ["CAPROV", "TAMAMU", "DENSIDAD", "SUPERF", 
                "AGUACALI", "CALEF", "ZONARES", "REGTEN", "NUMOCU",
                "NUMACTI", "NUMPERI", "NUMESTU", "NADUL_MAS", "NADUL_FEM",
                "NNINO_FEM", "NNINO_MAS", "OCUSP", "EDADSP", "NACION_ESP",
                "EDUC_SUPERIOR", "CAPROP", "CAJENA", "DISPOSIOV", "IMPEXAC",
                "Tmax_max", "Tmin_min", "Tasa_Paro", "Inflacion", "Tipo_Interes"]


# Helper inference

def infer(inputs_list):
    arr = np.array(inputs_list).reshape(1, -1)
    scaled = scaler_X.transform(arr)
    tensor = torch.tensor(scaled, dtype=torch.float32)
    with torch.no_grad():
        y_scaled = model(tensor).numpy().flatten()
    y = y_scaled * y_stds + y_means
    return {code: float(val) for code, val in zip(output_cols, y)}


simulator = TaxSimulator(
    infer_fn=infer,
    feature_cols=feature_cols,
    output_codes=output_cols,
    tax_csv_path='datamarts/tax_datamart_2025.tsv'
)









# --- Original predict route ---
@api.route('/predict', methods=['POST'])
def predict():
    data = request.get_json() or {}
    inputs = data.get('inputs')
    if not inputs or len(inputs) != input_size:
        return jsonify({"error": f"'inputs' must be a list of {input_size} values"}), 400
    preds = infer(inputs)
    return jsonify({"predictions": preds})

# --- New route: predict with child deduction ---
@api.route('/predict_child_effect', methods=['POST'])
def predict_child_effect():
    """
    Applies a fixed increase to net monthly income (IMPEXAC) per child,
    then returns both original and adjusted predictions.
    Expected JSON payload:
    {
      "inputs": [...],
      "deduction_per_child": 50.0  # optional, default 50
    }
    """
    data = request.get_json() or {}
    inputs = data.get('inputs')
    if not inputs or len(inputs) != input_size:
        return jsonify({"error": f"'inputs' must be a list of {input_size} values"}), 400
    # Deduction amount per child
    ded = float(data.get('deduction_per_child', 50.0))
    # Count children: indices 14 (female) + 15 (male)
    num_children = int(inputs[14]) + int(inputs[15])
    # Original predictions
    original = infer(inputs)
    # Adjust IMPEXAC at index 23
    adjusted_inputs = inputs.copy()
    adjusted_inputs[23] = float(adjusted_inputs[23]) + ded * num_children
    # Adjusted predictions
    adjusted = infer(adjusted_inputs)
    return jsonify({
        "original_predictions": original,
        "adjusted_predictions": adjusted,
        "num_children": num_children,
        "deduction_per_child": ded
    })



# --- Tax simulation route ---
@api.route('/simulate_tax', methods=['POST'])
def simulate_tax():
    """
    Expects JSON:
    {
      "inputs": [...],
      "deduction_per_child": float,
      "ccaa": int
    }
    Returns revenue estimation.
    """
    data = request.get_json() or {}
    inputs = data.get('inputs')
    ded = float(data.get('deduction_per_child', 50.0))
    ccaa = data.get('ccaa')

    if not inputs or len(inputs) != input_size or ccaa is None:
        return jsonify({"error": "Provide 'inputs', 'deduction_per_child' and 'ccaa'"}), 400

    # Llamamos al simulador global
    result = simulator.simulate(inputs, ded, int(ccaa))
    return jsonify(result)