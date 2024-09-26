# from flask import Flask, request, redirect, url_for, send_from_directory
# import os

# app = Flask(__name__)
# UPLOAD_FOLDER = 'uploads'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# @app.route('/')
# def upload_form():
#     return send_from_directory('', 'index.html')

# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'file' not in request.files:
#         return 'No file part'
#     file = request.files['file']
#     if file.filename == '':
#         return 'No selected file'
#     if file:
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
#         file.save(filepath)
#         return f'File successfully uploaded to {filepath}'

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, request, redirect, url_for
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'  # Replace with your actual folder path

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return 'No file part', 400  # Return a 400 Bad Request error code
    
    file = request.files['file']
    
    # If user does not select file, browser also submit an empty part without filename
    if file.filename == '':
        return 'No selected file', 400  # Return a 400 Bad Request error code
    
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        
        try:
            file.save(filepath)
            return f'File successfully uploaded to {filepath}', 200  # Return a 200 OK status code
        except Exception as e:
            return str(e), 500  # Return a 500 Internal Server Error code with the exception message

if __name__ == '__main__':
    app.run(debug=True)

