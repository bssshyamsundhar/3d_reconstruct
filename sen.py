from flask import Flask, request, send_file, render_template, redirect, url_for, jsonify
import os
from model import reconstruct
import open3d as o3d

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file'})
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        print("file saved successfully")
        # Process the file with Python
        mesh = reconstruct(file_path)
        # Save the processed file as mesh_with_texture.ply
        output_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'mesh_with_texture.ply')
        o3d.io.write_triangle_mesh(output_file_path, mesh)
        print("Mesh with texture saved")
        return jsonify({'status': 'success', 'message': f'File {file.filename} uploaded and processed successfully'})

@app.route('/get_ply')
def get_mesh():
    mesh_path = os.path.join(app.config['UPLOAD_FOLDER'], 'mesh_with_texture.ply')
    return send_file(mesh_path)

if __name__ == '__main__':
    app.run(debug=True)
