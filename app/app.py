import os
import uuid
from flask import Flask, render_template, request, jsonify, send_from_directory
from model_manager import ModelManager
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['GENERATED_FOLDER'] = 'static/generated'

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['GENERATED_FOLDER'], exist_ok=True)

model_manager = ModelManager()

@app.route('/')
def index():
    return render_template('layout.html')

@app.route('/vqa')
def vqa_page():
    return render_template('vqa.html')

@app.route('/generate')
def generate_page():
    return render_template('generate.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    if 'image' not in request.files and 'image_path' not in request.form:
        return jsonify({'error': 'No image provided'}), 400
    
    prompt = request.form.get('prompt', 'Describe this image.')
    model_size = request.form.get('model_size')
    
    image_path = None
    if 'image' in request.files:
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        filename = str(uuid.uuid4()) + "_" + file.filename
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(image_path)
    else:
        # Re-use existing image
        image_path = request.form.get('image_path')
        if not os.path.exists(image_path):
             return jsonify({'error': 'Image not found'}), 404

    try:
        response = model_manager.process_vqa(image_path, prompt, model_size)
        return jsonify({'response': response, 'image_path': image_path})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate', methods=['POST'])
def generate():
    prompt = request.form.get('prompt')
    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400
    
    try:
        image = model_manager.generate_image(prompt)
        filename = str(uuid.uuid4()) + ".png"
        filepath = os.path.join(app.config['GENERATED_FOLDER'], filename)
        image.save(filepath)
        return jsonify({'image_url': filepath})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/config', methods=['GET', 'POST'])
def config():
    if request.method == 'GET':
        return jsonify(model_manager.get_config())
    
    data = request.json
    if 'device' in data:
        try:
            model_manager.set_device(data['device'])
        except Exception as e:
            return jsonify({'error': str(e)}), 400
            
    return jsonify(model_manager.get_config())

@app.route('/api/logs', methods=['GET'])
def get_logs():
    return jsonify({'logs': model_manager.get_logs()})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
