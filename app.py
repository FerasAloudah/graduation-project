from flask import Flask, send_file, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from flask_restful import Resource, Api
from relationship_detector import *
import os

UPLOAD_FOLDER = 'images'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

api = Api(app)


@app.route('/')
def home():
    return render_template('index.html')


@app.route("/<string:file_name>")
def get_image(file_name):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], file_name))


class UploadImage(Resource):
    def post(self):
        if 'file' not in request.files:
            return 'Error'

        file = request.files['file']
        file_name = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
        file.save(file_path)

        predict_image(file_path)

        return file_name


api.add_resource(UploadImage, '/')

if __name__ == '__main__':
    app.run(debug=True)