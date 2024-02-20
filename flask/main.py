import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import io
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
import numpy as np
from PIL import Image

from flask import Flask, request, jsonify
from flask_cors import CORS



# ************************************************************ #

app = Flask(__name__)
CORS(app)

import pandas as pd

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        if file is None or file.filename == "":
            return jsonify({"error": "no file"})
        try:
            # Read the CSV file into a pandas DataFrame
            df = pd.read_csv(file)

            # Optionally, you can perform further processing or analysis on the DataFrame here
            # For example:
            # - Data cleaning
            # - Feature extraction
            # - Model prediction (if applicable)

            # Placeholder for model prediction result
            pred_labels = ["label1", "label2", "label3"]  # Dummy data

            data = {"prediction": pred_labels}
            return jsonify(data)
        except Exception as e:
            return jsonify({"error": str(e)})
    return "OK"



if __name__ == "__main__":
    app.run(debug=True)