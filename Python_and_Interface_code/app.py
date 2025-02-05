import os
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for
from utils import clean_data
from analysis import perform_analysis

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file and allowed_file(file.filename):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        try:
            # Read and clean the data
            df = pd.read_csv(filepath)
            df_cleaned = clean_data(df)

            # Perform clustering analysis
            plots = perform_analysis(df_cleaned)

            # Clean up uploaded file
            os.remove(filepath)
            return render_template('analysis.html', **plots)
        except Exception as e:
            return f"Error during processing: {e}"
    else:
        return "Invalid file format"


if __name__ == '__main__':
    app.run(debug=True)
