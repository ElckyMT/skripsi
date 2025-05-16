import pickle
import pandas as pd
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model
with open("model_random_forest_mahasiswa.pkl", "rb") as f:
    model = pickle.load(f)

# Label
LABEL = {0: "Terlambat", 1: "Tepat Waktu"}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_manual', methods=['POST'])
def predict_manual():
    try:
        ipk = float(request.form['ipk'])
        bobot = float(request.form['bobot'])
        sks = float(request.form['jumlah_sks_tempuh'])

        data = [[ipk, bobot, sks]]
        result = model.predict(data)[0]
        label = LABEL[result]

        return f'<h3>Hasil Prediksi: {label} ({result})</h3><br><a href="/">Kembali</a>'
    except Exception as e:
        return f"Terjadi error: {str(e)}"

@app.route('/predict_excel', methods=['POST'])
def predict_excel():
    try:
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            df = pd.read_excel(filepath)
            required_columns = ['ipk', 'bobot', 'jumlah_sks_tempuh']

            if not all(col in df.columns for col in required_columns):
                return f"File Excel harus memiliki kolom: {required_columns}"

            data = df[required_columns]
            pred = model.predict(data)

            df['prediksi'] = pred
            df['label'] = df['prediksi'].map(LABEL)

            result_html = df.to_html(index=False)

            return f'<h3>Hasil Prediksi Excel</h3>{result_html}<br><a href="/">Kembali</a>'
        else:
            return "File tidak ditemukan."
    except Exception as e:
        return f"Terjadi error saat memproses Excel: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
