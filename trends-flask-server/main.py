import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import io

from flask import Flask, request, jsonify
from flask_cors import CORS

# ************************************************************ #

app = Flask(__name__)
CORS(app)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras


def encode(val):
    le = LabelEncoder()
    return le.fit_transform(val)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "original" not in request.files or "anonymous" not in request.files:
            return jsonify({"error": "Incomplete or no files"})

        file1 = request.files["original"]
        file2 = request.files["anonymous"]
        data = request.form.to_dict()

        try:
            df1 = pd.read_csv(file1)
            df2 = pd.read_csv(file2)
            target = data["target"]

            # ************************************************************ #

            # We need to split original dataset into train and test
            X = df1.drop([target], axis=1)
            Y = df1[target]

            # Preprocess the data
            if isinstance(Y[0], str):
                Y = encode(Y)

            checkarr = X.iloc[1, :]
            for i, value in checkarr.iteritems():
                if isinstance(value, str):
                    X[i] = X[i].astype("category").cat.codes

            scaler1 = StandardScaler()
            X = scaler1.fit_transform(X)

            Y = Y.reshape(-1, 1)

            # ************************************************************ #

            x_train, x_test, y_train, y_test = train_test_split(
                X, Y, test_size=0.2, random_state=42
            )

            # ************************************************************ #

            X_synth = df2.drop([target], axis=1)
            Y_synth = df2[target]

            if isinstance(Y_synth[0], str):
                Y_synth = encode(Y_synth)

            checkarr = X_synth.iloc[1, :]
            for i, value in checkarr.iteritems():
                if isinstance(value, str):
                    X_synth[i] = X_synth[i].astype("category").cat.codes

            scaler2 = StandardScaler()
            X = scaler2.fit_transform(X)
            X_synth = scaler2.fit_transform(X_synth)

            # ************************************************************ #

            tf.random.set_seed(3)

            model1 = keras.Sequential(
                [
                    keras.layers.Flatten(input_shape=X[0].shape),
                    keras.layers.Dense(20, activation="relu"),
                    keras.layers.Dense(2, activation="sigmoid"),
                ]
            )

            model1.compile(
                optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )

            history = model1.fit(x_train, y_train, epochs=10)

            # ************************************************************ #
            model2 = keras.Sequential(
                [
                    keras.layers.Flatten(input_shape=X[0].shape),
                    keras.layers.Dense(20, activation="relu"),
                    keras.layers.Dense(2, activation="sigmoid"),
                ]
            )

            model2.compile(
                optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )

            history_synth = model2.fit(X_synth, Y_synth, epochs=10)

            # ************************************************************ #

            acc1, loss1 = model1.evaluate(x_test, y_test)
            acc2, loss2 = model2.evaluate(X_synth, Y_synth)

            # ************************************************************ #

            results = {
                "original": {"accuracy": acc1, "loss": loss1},
                "anonymous": {"accuracy": acc2, "loss": loss2},
            }

            return jsonify(results)
        except Exception as e:
            return jsonify({"error": str(e)})
    return "OK"


if __name__ == "__main__":
    app.run(debug=True)
