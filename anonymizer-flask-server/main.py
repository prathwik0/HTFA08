import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import io
import numpy as np

from flask import Flask, request, jsonify
from flask_cors import CORS

# ************************************************************ #

app = Flask(__name__)
CORS(app)

import pandas as pd
from ctgan import CTGAN


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        if file is None or file.filename == "":
            return jsonify({"error": "no file"})
        try:
            df = pd.read_csv(file)
            columns = df.columns.tolist()

            ctgan = CTGAN(epochs=10)
            ctgan.fit(df, columns)

            synthetic_data = ctgan.sample(1000)

            csv_string = io.StringIO()
            synthetic_data.to_csv(csv_string, index=False)
            csv_string.seek(0)

            return jsonify(csv_string.getvalue())
        except Exception as e:
            return jsonify({"error": str(e)})
    return "OK"


if __name__ == "__main__":
    app.run(debug=True)
