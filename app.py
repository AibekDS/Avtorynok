from flask import Flask, render_template, request, send_from_directory
from main_code import predict_num
import os
import cv2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './images/'
app.config['PROCESSED_FOLDER'] = './processed/'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

# best = "best-15.pt"
best = "best_large_params.pt"


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], imagefile.filename)
    imagefile.save(image_path)
    
    # Run the prediction and save the final processed image
    final_img = predict_num(image_path, best)
    processed_image_path = os.path.join(app.config['PROCESSED_FOLDER'], "processed_"+imagefile.filename)
    cv2.imwrite(processed_image_path, cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))

    # Pass the path to the template
    return render_template('index.html', prediction_image=processed_image_path)

@app.route('/processed/<filename>')
def processed_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

if __name__ == '__main__':
    app.run(port=3000, debug=True)
