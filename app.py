from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory
import cv2
import numpy as np
import tensorflow as tf
import os
import uuid
import json
from datetime import datetime
import time
import logging
import sys
import base64
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Set TensorFlow to be deterministic for consistent results
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
tf.config.experimental.enable_op_determinism()

# Force UTF-8 encoding for console logs to avoid emoji encode errors on Windows consoles
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()])
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
IMAGE_UPLOAD_FOLDER = os.path.join(UPLOAD_FOLDER, 'images')
app.config['IMAGE_UPLOAD_FOLDER'] = IMAGE_UPLOAD_FOLDER
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp', 'tiff'}

# Create necessary directories
os.makedirs(IMAGE_UPLOAD_FOLDER, exist_ok=True)
os.makedirs('model', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# Model paths - Updated to match your actual model files
IMAGE_MODEL_PATH = os.path.join('model', 'final_xception.keras')

# Global variables to store the models
image_model = None

def load_image_model():
    global image_model
    if os.path.exists(IMAGE_MODEL_PATH):
        try:
            # Try loading with custom objects if needed
            logger.info(f"Attempting to load image model from {IMAGE_MODEL_PATH}")
            
            # First try normal loading
            image_model = tf.keras.models.load_model(IMAGE_MODEL_PATH)
            logger.info("Image model loaded successfully!")
            
            # Test the model with a dummy input to ensure it works
            dummy_input = np.random.random((1, 299, 299, 3)).astype(np.float32)
            test_prediction = image_model.predict(dummy_input, verbose=0)
            logger.info(f"Model test successful. Output shape: {test_prediction.shape}")
            
            return True
        except Exception as e:
            logger.error(f"Error loading image model: {e}")
            try:
                # Fallback: Load with compile=False
                logger.info("Trying to load with compile=False...")
                image_model = tf.keras.models.load_model(IMAGE_MODEL_PATH, compile=False)
                
                # Test the model
                dummy_input = np.random.random((1, 299, 299, 3)).astype(np.float32)
                test_prediction = image_model.predict(dummy_input, verbose=0)
                logger.info(f"Image model loaded with compile=False. Output shape: {test_prediction.shape}")
                return True
            except Exception as e2:
                logger.error(f"Failed to load image model even with fallback: {e2}")
                return False
    else:
        logger.warning(f"Image model file not found at {IMAGE_MODEL_PATH}. Using demo mode.")
        return False

# Try to load the models at startup
logger.info("Loading models at startup...")
image_model_loaded = load_image_model()

# Log the final status
if image_model_loaded:
    logger.info("IMAGE MODEL: Successfully loaded and ready for predictions")
else:
    logger.error("IMAGE MODEL: Failed to load - will use demo mode")

def allowed_image_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS

def preprocess_image_for_model(image, target_size=(299, 299)):
    """
    Preprocess image for model prediction with proper normalization
    """
    try:
        # Ensure image is in the right format
        if image is None:
            raise ValueError("Input image is None")
        # Convert to RGB if needed
        if len(image.shape) == 3:
            if image.shape[2] == 4:  # RGBA
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            elif image.shape[2] == 3:  # BGR
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        # Resize to target size
        image = cv2.resize(image, target_size)
        # Ensure the image is float32 and normalized to [0, 1]
        image = image.astype(np.float32)
        if image.max() > 1.0:
            image = image / 255.0
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        logger.info(f"Preprocessed image shape: {image.shape}, dtype: {image.dtype}, range: [{image.min():.3f}, {image.max():.3f}]")
        return image
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        return None

def predict_image(image):
    global image_model, image_model_loaded
    
    # Log whether we're using real model or demo mode
    if image_model_loaded and image_model is not None:
        logger.info("Using REAL deepfake detection model for prediction")
        logger.info(f"Model type: {type(image_model).__name__}")
        print("🔍 Using REAL deepfake detection model for prediction")
    else:
        logger.warning("WARNING: No real model loaded, will use demo mode!")
        logger.info("Using DEMO mode for prediction (no real model loaded)")
        print("⚠️  WARNING: Using DEMO mode (no real model loaded)")
    
    # If model is not loaded, try to load it again
    if not image_model_loaded:
        logger.info("Attempting to reload image model...")
        image_model_loaded = load_image_model()
        
    if not image_model_loaded or image_model is None:
        # If we don't have a real model, use a demo mode with simulated results
        logger.warning("No image model loaded, using demo mode with simulated results")
        return simulate_prediction(image)
    
    try:
        processed_image = preprocess_image_for_model(image)
        if processed_image is None:
            return "Error", 0.0, "Failed to preprocess image"
            
        # Add a small delay to simulate processing time
        time.sleep(0.5)
        
        logger.info("Running REAL model prediction...")
        print("🤖 Running REAL model prediction...")
        
        # Get prediction from the model
        raw_prediction = image_model.predict(processed_image, verbose=0)
        logger.info(f"Raw prediction from REAL model: {raw_prediction}")
        print(f"📊 Raw prediction: {raw_prediction}")
        
        # Handle different output formats
        if len(raw_prediction.shape) > 1 and raw_prediction.shape[1] > 1:
            # Multi-class output
            fake_prob = raw_prediction[0][1] if raw_prediction.shape[1] > 1 else raw_prediction[0][0]
        else:
            # Binary output
            fake_prob = raw_prediction[0][0]
        
        # Determine result
        result = "Fake" if fake_prob > 0.5 else "Real"
        confidence = fake_prob * 100 if result == "Fake" else (1 - fake_prob) * 100
        confidence = max(50.0, min(95.0, confidence))
        
        # Log the result prominently
        logger.info("=" * 50)
        logger.info(f"REAL MODEL PREDICTION RESULT: {result}")
        logger.info(f"CONFIDENCE: {confidence:.1f}%")
        logger.info(f"Fake probability: {fake_prob:.4f}")
        logger.info(f"Real probability: {1-fake_prob:.4f}")
        logger.info("=" * 50)
        
        # Also print to terminal immediately
        print("=" * 50)
        print(f"🎯 REAL MODEL PREDICTION RESULT: {result}")
        print(f"📈 CONFIDENCE: {confidence:.1f}%")
        print(f"🔴 Fake probability: {fake_prob:.4f}")
        print(f"🟢 Real probability: {1-fake_prob:.4f}")
        print("=" * 50)
        
        # Force flush the logging buffer
        import sys
        sys.stdout.flush()
        
        return result, confidence, f"Analysis completed with {confidence:.1f}% confidence"
        
    except Exception as e:
        error_msg = f"Error during prediction: {str(e)}"
        logger.error(error_msg)
        print(f"❌ ERROR: {error_msg}")
        return "Error", 0.0, error_msg

def simulate_prediction(image):
    """Simulate a prediction when no model is available"""
    try:
        # Calculate image characteristics for more realistic simulation
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate various image quality metrics
        blur_variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        brightness = np.mean(gray)
        contrast = gray.std()
        
        # Normalize metrics
        blur_factor = min(blur_variance / 1000, 1.0)
        brightness_factor = abs(brightness - 128) / 128
        contrast_factor = min(contrast / 50, 1.0)
        
        # Calculate a deterministic probability (remove random component)
        # Use a hash of the image characteristics for consistent results
        image_hash = hash((int(blur_variance), int(brightness), int(contrast)))
        
        # Combine factors to determine if image appears fake
        manipulation_score = (
            (1 - blur_factor) * 0.3 +
            (1 - brightness_factor) * 0.3 +
            contrast_factor * 0.2 +
            (image_hash % 100) / 100 * 0.2  # Deterministic instead of random
        )
        
        # Determine result
        if manipulation_score > 0.6:
            result = "Real"
            confidence = 60 + (manipulation_score - 0.6) * 100
        else:
            result = "Fake"
            confidence = 60 + (0.6 - manipulation_score) * 100
        
        confidence = min(confidence, 95)
        
        logger.info(f"Demo mode - Result: {result}, Confidence: {confidence:.1f}%")
        return result, confidence, "Demo mode: analysis based on image characteristics"
    except Exception as e:
        logger.error(f"Error in simulation: {e}")
        return "Real", 75.0, "Demo mode: fallback result"

def log_prediction(file_path, result, confidence, file_type="image"):
    """Log prediction to a file for analytics"""
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, 'predictions.json')
    
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'file_path': file_path,
        'file_type': file_type,
        'result': result,
        'confidence': float(confidence)
    }
    
    # Load existing logs
    logs = []
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r') as f:
                logs = json.load(f)
        except Exception as e:
            logger.error(f"Error reading log file: {e}")
            logs = []
    
    # Add new log
    logs.append(log_entry)
    
    # Keep only last 1000 entries
    if len(logs) > 1000:
        logs = logs[-1000:]
    
    # Save logs
    try:
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)
        logger.info(f"Logged prediction for {file_path}")
    except Exception as e:
        logger.error(f"Error writing to log file: {e}")

# Grad-CAM utility function

def generate_gradcam_heatmap(model, image_array, last_conv_layer_name=None, pred_index=None):
    """
    Generate a Grad-CAM heatmap for a given image and model.
    Args:
        model: The Keras model.
        image_array: Preprocessed image array (batch, h, w, c).
        last_conv_layer_name: Name of the last conv layer (if None, auto-detects).
        pred_index: Index of the class to visualize (if None, uses model prediction).
    Returns:
        heatmap: 2D numpy array (h, w) normalized to [0, 1].
    """
    import tensorflow as tf
    # Find the last conv layer if not provided
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if 'conv' in layer.name and len(layer.output_shape) == 4:
                last_conv_layer_name = layer.name
                break
        else:
            raise ValueError("No convolutional layer found in model.")
    last_conv_layer = model.get_layer(last_conv_layer_name)
    grad_model = tf.keras.models.Model(
        [model.inputs], [last_conv_layer.output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Overlay heatmap on image

def overlay_heatmap_on_image(original_img, heatmap, alpha=0.4, colormap=cm.jet):
    """
    Overlay a heatmap onto an image.
    Args:
        original_img: Original image (H, W, 3), uint8 or float32.
        heatmap: 2D numpy array (H, W) normalized to [0, 1].
        alpha: Transparency factor.
        colormap: Matplotlib colormap.
    Returns:
        overlayed_img: uint8 image with heatmap overlay.
    """
    import cv2
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = colormap(heatmap)
    heatmap_color = np.uint8(heatmap_color[:, :, :3] * 255)
    overlayed_img = cv2.addWeighted(original_img, 1 - alpha, heatmap_color, alpha, 0)
    return overlayed_img

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/image-detection')
def image_detection():
    return render_template('image_detection.html')

@app.route('/upload-image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if not file or not file.filename:
        return jsonify({'error': 'No selected file'})
    if file and allowed_image_file(file.filename):
        try:
            file_ext = file.filename.rsplit('.', 1)[1].lower()
            unique_filename = f"{uuid.uuid4().hex}.{file_ext}"
            filepath = os.path.join(app.config['IMAGE_UPLOAD_FOLDER'], unique_filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            file.save(filepath)
            logger.info(f"Saved uploaded image to {filepath}")
            image = cv2.imread(filepath)
            if image is None:
                return jsonify({'error': 'Failed to read image'})
            result, confidence, message = predict_image(image)
            if result == "Error":
                return jsonify({'error': f'Analysis error: {message}'})
            log_prediction(filepath, result, confidence, "image")
            return jsonify({
                'result': result,
                'confidence': f"{confidence:.1f}%",
                'image_url': url_for('static', filename=f'uploads/images/{unique_filename}'),
                'message': message,
                'type': 'image'
            })
        except Exception as e:
            logger.error(f"Error processing image upload: {e}")
            return jsonify({'error': f'Processing error: {str(e)}'})
    return jsonify({'error': 'Invalid file format'})


@app.route('/webcam', methods=['POST'])
def webcam_upload():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data'})
        
        # Decode base64 image
        img_data = base64.b64decode(data['image'].split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Failed to decode image'})
        
        # Generate a unique filename
        unique_filename = f"{uuid.uuid4().hex}.jpg"
        filepath = os.path.join(app.config['IMAGE_UPLOAD_FOLDER'], unique_filename)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the image
        cv2.imwrite(filepath, image)
        logger.info(f"Saved webcam image to {filepath}")
        
        # Predict
        result, confidence, message = predict_image(image)
        
        if result == "Error":
            return jsonify({'error': f'Analysis error: {message}'})
        
        # Log the prediction
        log_prediction(filepath, result, confidence, "webcam")
        
        return jsonify({
            'result': result,
            'confidence': f"{confidence:.1f}%",
            'image_url': url_for('static', filename=f'uploads/images/{unique_filename}'),
            'message': message,
            'type': 'image'
        })
    except Exception as e:
        logger.error(f"Error processing webcam image: {e}")
        return jsonify({'error': f'Processing error: {str(e)}'})

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring"""
    return jsonify({
        'status': 'ok',
        'image_model_loaded': image_model_loaded,
        'image_model_path': IMAGE_MODEL_PATH,
        'image_model_exists': os.path.exists(IMAGE_MODEL_PATH),
        'tensorflow_version': tf.__version__,
        'timestamp': datetime.now().isoformat()
    })

# Static file serving routes
@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files"""
    return send_from_directory('static', filename)

@app.route('/static/uploads/images/<filename>')
def serve_image(filename):
    """Serve uploaded images"""
    return send_from_directory(app.config['IMAGE_UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    logger.info("Starting DeepFake Detector application")
    logger.info(f"TensorFlow version: {tf.__version__}")
    logger.info(f"Image model path: {IMAGE_MODEL_PATH}")
    logger.info(f"Image model exists: {os.path.exists(IMAGE_MODEL_PATH)}")
    app.run(debug=True, host='0.0.0.0', port=5000)
