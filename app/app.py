from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf

app = Flask(__name__)
model = tf.keras.models.load_model("../ocr_model.keras")

# Same label mapping used during training
unique_labels = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57,  # 0–9
                 65, 66, 67, 68, 69, 70, 71, 72, 73, 74,  # A–J
                 75, 76, 77, 78, 79, 80, 81, 82, 83, 84,  # K–T
                 85, 86, 87, 88, 89, 90]                  # U–Z

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    matrix = np.array(data["grid"])

    # Extract features: column sums + row sums
    col_sums = matrix.sum(axis=0)
    row_sums = matrix.sum(axis=1)
    features = np.concatenate([col_sums, row_sums]).reshape(1, -1)

    prediction = model.predict(features, verbose=0)
    pred_index = np.argmax(prediction)
    pred_ascii = unique_labels[pred_index]
    pred_char = chr(pred_ascii)

    return jsonify({"ascii": pred_ascii, "char": pred_char})

if __name__ == "__main__":
    app.run(debug=True)
