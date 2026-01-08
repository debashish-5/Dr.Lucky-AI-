import os, time, math
from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model

# ---------------- CONFIG ----------------
UPLOAD_FOLDER = 'static/images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXT = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ---------------- LOAD MODEL ----------------
def safe_load(path):
    try:
        m = load_model(path)
        print(f"Loaded model: {path}, input_shape={m.input_shape}")
        return m
    except Exception as e:
        print(f"Warning: couldn't load {path}: {e}")
        return None

xray_model = safe_load('model_101.h5')
XRAY_LABELS  = ['Normal', 'Pneumonia']

# ---------------- UTILS ----------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXT

def preprocess_image_to_model(path, model):
    """Prepare image to match model input size and channels"""
    shape = model.input_shape  # e.g., (None, 100,100,1)
    
    if len(shape) == 4:
        _, h, w, c = shape
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE if c==1 else cv2.IMREAD_COLOR)
        if c!=1: 
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (w, h))
        arr = img.astype("float32") / 255.0
        arr = arr.reshape(1, h, w, c)
        return arr
    else:
        raise ValueError(f"Unsupported model input shape: {shape}")

def predict_and_confidences(model, image):
    """Return (no_conf, yes_conf, raw)"""
    raw = model.predict(image)
    raw = np.array(raw)
    if raw.ndim==2 and raw.shape[1]>=2:  # softmax
        probs = raw[0]/raw[0].sum()
        no_conf = float(probs[0])
        yes_conf = float(probs[1])
    else:  # sigmoid
        val = float(raw[0][0])
        val = max(0.0, min(1.0, val))
        no_conf = 1.0 - val
        yes_conf = val
    return no_conf, yes_conf, raw

# ---------------- VISUALS ----------------
def save_bar(no, yes, out_path):
    labels = ['Normal','Pneumonia']
    values = [no*100, yes*100]
    fig, ax = plt.subplots(figsize=(6,3.2), dpi=120)
    bars = ax.bar(labels, values, color=['#60a5fa','#2dd4bf'], edgecolor='black', linewidth=0.6)
    ax.set_ylim(0,100)
    ax.set_ylabel('Confidence %')
    ax.set_title('Prediction Confidence')
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, v + 1.5, f'{v:.1f}%', ha='center', weight='bold')
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def make_visuals(img_path, prepared, no, yes, prefix):
    ts = str(int(time.time()*1000))
    bar = os.path.join(UPLOAD_FOLDER, f"{prefix}_bar_{ts}.png")
    save_bar(no, yes, bar)
    return {'bar': bar}

# ---------------- ROUTES ----------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/xray', methods=['GET','POST'])
def xray():
    if request.method=='POST':
        file = request.files.get('image')
        if not file or file.filename=='': 
            return render_template('xray.html', error="Choose an image")
        if not allowed_file(file.filename): 
            return render_template('xray.html', error="Allowed types: png,jpg,jpeg")
        
        filename = secure_filename(str(int(time.time()))+"_"+file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)
        
        if xray_model is None: 
            return render_template('xray.html', error="X-ray model not loaded")
        
        img = preprocess_image_to_model(save_path, xray_model)
        no_conf, yes_conf, raw = predict_and_confidences(xray_model, img)
        visuals = make_visuals(save_path, img, no_conf, yes_conf, 'xray')
        chart_urls = {k:url_for('static', filename=f'images/{os.path.basename(v)}') for k,v in visuals.items()}
        
        # Determine result
        if yes_conf >= 0.7: 
            result_label = "Pneumonia Detected"
        elif yes_conf <= 0.3: 
            result_label = "Normal"
        else: 
            result_label = "Uncertain – Clinical Review Recommended"

        return render_template('result.html',
                               model_name='Chest X-ray',
                               result=result_label,
                               image=url_for('static', filename=f'images/{filename}'),
                               charts=chart_urls, no_conf=no_conf, yes_conf=yes_conf)
    return render_template('xray.html')

if __name__=='__main__':
    app.run(debug=True)
