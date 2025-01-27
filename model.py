import os
import torch
from PIL import Image
from find_truck_license_plate import find_truck_license_plate
from preprocessing_plate import preprocessing_plate

from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__, static_folder='static')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

UPLOAD_FOLDER = os.path.join('static', 'uploads') 
DETECTED_FOLDER = os.path.join('static', 'detected')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg','PNG','JPG','JPEG'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DETECTED_FOLDER'] = DETECTED_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_license_plate_text(image_files, find_truck_license_plate, device):
    license_plate_detection, result = find_truck_license_plate(image_files, device) 
    filtered_text = []
    for (_, text, _) in result:
        filtered = ''.join([char for char in text if char.isdigit() or ('가' <= char <= '힣')])
        if filtered:  # 빈 문자열은 추가하지 않도록
            filtered_text.append(filtered)
    result =  ''.join(filtered_text)
    license_plate_detection, result = preprocessing_plate(license_plate_detection, device)

    base_filename = os.path.splitext(os.path.basename(image_files[0]))[0]
    detected_filename = f"{base_filename}_detected.png"
    detected_path = os.path.join(app.config['DETECTED_FOLDER'], detected_filename)

    # NumPy 배열을 PIL Image로 변환
    pil_image = Image.fromarray(license_plate_detection.astype('uint8'))
      
    pil_image.save(detected_path)
    detected_image_url = os.path.join(app.config['DETECTED_FOLDER'], detected_filename)

    return detected_image_url, result

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            detected_image_url, result = extract_license_plate_text([file_path], find_truck_license_plate, device)
            
            # 결과가 None이 아닌 경우에만 렌더링
            if detected_image_url is not None and result is not None:
               return render_template('result.html', original_image_path=file_path, detected_image_path=detected_image_url, result=result)
            else:
                return "License plate not detected or error occurred"

    return render_template('index.html')

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(DETECTED_FOLDER, exist_ok=True)
    app.run(debug=True)


