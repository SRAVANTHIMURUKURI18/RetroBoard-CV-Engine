import os
import uuid
import cv2
import numpy as np
import faiss
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- FAISS DATABASE SETUP ---
vector_dimension = 3 
faiss_index = faiss.IndexFlatL2(vector_dimension)
image_database = {} 
current_index_id = 0

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_color_vector(image_path):
    img = cv2.imread(image_path)
    if img is None: return None
    avg_color_per_row = np.average(img, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    return np.array([avg_color], dtype=np.float32)

def create_lut(x, y):
    """Helper for Cinematic LUTs"""
    return np.interp(np.arange(256), x, y).astype(np.uint8)

def add_grain_and_vignette(img, intensity=10):
    """Helper function to add authentic film grain and darkened edges."""
    rows, cols, ch = img.shape
    
    # Grain
    noise = np.random.normal(0, intensity, (rows, cols, ch)).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255)
    
    # Vignette
    kernel_x = cv2.getGaussianKernel(cols, cols / 1.5)
    kernel_y = cv2.getGaussianKernel(rows, rows / 1.5)
    kernel = kernel_y * kernel_x.T
    mask = kernel / kernel.max()
    img = img * np.dstack([mask] * 3)
    
    return np.clip(img, 0, 255).astype(np.uint8)

def apply_filter(image_path, output_path, style):
    """Routes the image to the correct mathematical filter based on the dropdown."""
    img = cv2.imread(image_path)
    if img is None: return False

    if style == 'bw':
        # Classic B&W Grunge (High contrast, heavy grain)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.convertScaleAbs(gray, alpha=1.3, beta=-20) 
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        img = add_grain_and_vignette(img, intensity=25)

    elif style == 'warm':
        # Golden Hour Warmth (Boost reds/greens, drop blues)
        img_float = img.astype(np.float32)
        b, g, r = cv2.split(img_float)
        r = np.clip(r * 1.2 + 10, 0, 255)
        g = np.clip(g * 1.1 + 5, 0, 255)
        b = np.clip(b * 0.8, 0, 255)
        img = cv2.merge((b, g, r))
        img = cv2.convertScaleAbs(img, alpha=0.9, beta=20)
        img = add_grain_and_vignette(img, intensity=8)

    elif style == 'retro':
        # 90s Retro Cross-Process (Magenta shadows, yellow highlights)
        img_float = img.astype(np.float32)
        b, g, r = cv2.split(img_float)
        r = np.clip(r * 1.1, 0, 255)
        g = np.clip(g * 0.9, 0, 255)
        b = np.clip(b * 1.1 + 15, 0, 255) 
        img = cv2.merge((b, g, r)).astype(np.uint8)
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= 0.7 # Desaturate slightly
        img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
        img = add_grain_and_vignette(img, intensity=15)

    else: 
        # Cinematic Vintage Cam (The high-end Bloom + LUTs you loved)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        _, bright_mask = cv2.threshold(v, 180, 255, cv2.THRESH_BINARY)
        bright_areas = cv2.bitwise_and(img, img, mask=bright_mask)
        bloom = cv2.GaussianBlur(bright_areas, (0, 0), 25)
        img = cv2.addWeighted(img, 1.0, bloom, 0.4, 0)

        r_x, r_y = [0, 64, 128, 192, 255], [0, 50, 135, 210, 255]
        g_x, g_y = [0, 64, 128, 192, 255], [0, 55, 128, 205, 255]
        b_x, b_y = [0, 64, 128, 192, 255], [30, 70, 120, 180, 230] 

        b, g, r = cv2.split(img)
        r = cv2.LUT(r, create_lut(r_x, r_y))
        g = cv2.LUT(g, create_lut(g_x, g_y))
        b = cv2.LUT(b, create_lut(b_x, b_y))
        img = cv2.merge((b, g, r))

        img = cv2.addWeighted(img, 0.95, np.full(img.shape, 25, dtype=np.uint8), 0.05, 0)
        img = add_grain_and_vignette(img, intensity=8)

    cv2.imwrite(output_path, img)
    return True

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    global current_index_id
    
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
            
        file = request.files['file']
        filter_style = request.form.get('filter_style', 'vintage_cam')
        
        filter_names = {
            'vintage_cam': 'Cinematic Vintage Cam',
            'warm': 'Golden Hour Warmth',
            'retro': '90s Retro',
            'bw': 'Classic B&W Grunge'
        }
        
        if file.filename == '' or not allowed_file(file.filename):
            return redirect(request.url)
            
        if file:
            unique_id = str(uuid.uuid4().hex)[:8]
            ext = file.filename.rsplit('.', 1)[1].lower()
            original_filename = f"orig_{unique_id}.{ext}"
            processed_filename = f"pin_{unique_id}.{ext}"
            
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
            processed_filepath = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
            
            file.save(filepath)
            success = apply_filter(filepath, processed_filepath, filter_style)
            
            if not success:
                return "Error processing image", 500

            # FAISS LOGIC
            similar_images = []
            vector = extract_color_vector(processed_filepath)
            
            if vector is not None:
                if faiss_index.ntotal > 0:
                    distances, indices = faiss_index.search(vector, min(3, faiss_index.ntotal))
                    for idx in indices[0]:
                        if idx in image_database:
                            similar_images.append(image_database[idx])
                
                faiss_index.add(vector)
                image_database[current_index_id] = processed_filename
                current_index_id += 1
            
            return render_template('index.html', 
                                   original_image=original_filename, 
                                   processed_image=processed_filename,
                                   applied_filter=filter_names.get(filter_style, 'Edit'),
                                   similar_images=similar_images)
                                   
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)