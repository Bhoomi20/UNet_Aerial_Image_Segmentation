import os
import time
from flask import Flask, render_template, request,jsonify
from flask_cors import CORS
from Main_Test_Code import startProcessing
from flask import make_response
from functools import wraps, update_wrapper
from datetime import datetime


app = Flask(__name__)
CORS(app)
app.config['UPLOAD_DIR'] = os.path.join(os.getcwd(), 'upload')
app.config['RESULT_DIR'] = os.path.join(os.getcwd(), './static/results')
app.config['ALLOWED_EXTENSIONS'] = set(['jpg', 'jpeg', 'png'])
app.config["CACHE_TYPE"] = "null"

# Setup
if not os.path.exists(app.config['UPLOAD_DIR']):
    os.mkdir(app.config['UPLOAD_DIR'])

if not os.path.exists(app.config['RESULT_DIR']):
    os.mkdir(app.config['RESULT_DIR'])

# Helpers
def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

def error(msg):
    return jsonify({'error': msg})

def nocache(view):
    @wraps(view)
    def no_cache(*args, **kwargs):
        response = make_response(view(*args, **kwargs))
        response.headers['Last-Modified'] = datetime.now()
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '-1'
        return response
        
    return update_wrapper(no_cache, view)

@app.route('/')
def upload_file():
   return render_template('index.html')
	
@app.route('/uploader', methods = ['GET', 'POST'])
@nocache
def post_uploader():
   if request.method == 'POST':
    image = request.files['pre-disaster']
    image1 = request.files['post-disaster']
    if not allowed_file(image.filename):
        return error('Only supported %s' % app.config['ALLOWED_EXTENSIONS']), 415
    
    t = int(time.time())
    image_dir = os.path.join(app.config['UPLOAD_DIR'], str(t))
    image_path = os.path.join(image_dir, "Pre.png")
    image_path1 = os.path.join(image_dir, "Post.png")

    os.mkdir(image_dir)
    image.save(image_path)
    image1.save(image_path1)
    #return "Result will be shown here!"
    data = startProcessing(image_path,image_path1)
    return render_template('result.html',value = round(data[0]),input_image_path = data[1],Enhance_Image_path = data[2],segmented_Image_path = data[3],mark_damage_region_path = data[4] )
		
if __name__ == '__main__':
   app.run(debug = True)