from flask import Flask, request, jsonify, render_template
from enhanced_color_model import EnhancedColorModel
import os

app = Flask(__name__)

# Initialize the color model
model = EnhancedColorModel()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train_model():
    try:
        history = model.train(epochs=50)
        return jsonify({'status': 'success', 'message': 'Model trained successfully'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/generate', methods=['POST'])
def generate_palette():
    try:
        data = request.get_json()
        emotion = data.get('emotion')
        brief = data.get('brief')
        
        if not emotion or not brief:
            return jsonify({'status': 'error', 'message': 'Both emotion and brief are required'})
        
        palette = model.generate_palette(emotion, brief)
        return jsonify({
            'status': 'success',
            'palette': palette,
            'emotion': emotion,
            'brief': brief
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
    app.run(debug=True, port=5000)