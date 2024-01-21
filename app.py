from flask import Flask, jsonify, request
import socket
import os
from utils_model import *
import numpy as np

MODEL_DIR = os.path.join(".", "models")
LOG_DIR = os.path.join(".", "logs")

app = Flask(__name__)

@app.route("/")
def hello():
    name = os.getenv("NAME", "world")
    hostname = socket.gethostname()
    return f"<h3>Hello {name}!</h3><b>Hostname:</b> {hostname}<br/>"

def parse_request(request_data, required_fields):
    missing_fields = [field for field in required_fields if field not in request_data]
    if missing_fields:
        print(f"ERROR API: received request, but no {', '.join(missing_fields)} found within")
        return None
    return {field: request_data[field] for field in required_fields}

@app.route('/predict', methods=['GET','POST'])
def predict():
    required_fields = ['country', 'year', 'month', 'day', 'dev', 'verbose']
    data = parse_request(request.json, required_fields)
    if data is None:
        return jsonify([])

    data['dev'] = data['dev'] == "True"
    data['verbose'] = data.get('verbose', 'True') == "True"

    _result = model_predict(**data)
    result = {key: item.tolist() if isinstance(item, np.ndarray) else item for key, item in _result.items()}
    return jsonify(result)

@app.route('/train', methods=['GET','POST'])
def train():
    required_fields = ['dev', 'verbose']
    data = parse_request(request.json, required_fields)
    if not request.json:
        print("ERROR: API (train): did not receive request data")
        return jsonify(False)
    
    data['dev'] = data['dev'] == "True"
    data['verbose'] = data.get('verbose', 'True') == "True"
    
    #model = model_train(**data)
    model = model_load()
    print("... training complete")
    return jsonify(True)

@app.route('/logging', methods=['GET','POST'])
def load_logs():
    required_fields = ['env', 'verbose', 'tag', 'month', 'year']
    data = parse_request(request.json, required_fields)
    if data is None:
        return jsonify([])

    data['verbose'] = data['verbose'] == "True"

    logfile = log_load(**data)
    return jsonify({"logfile": logfile})

if __name__ == '__main__':
    app.run(debug=os.getenv("FLASK_DEBUG", False), host='0.0.0.0', port=8080)
