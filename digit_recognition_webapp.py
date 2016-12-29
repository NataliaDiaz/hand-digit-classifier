# -*- coding:utf-8 -*-
from flask import Flask, request
from flask import redirect, url_for
from werkzeug.utils import secure_filename
from sklearn.externals import joblib
# joblib.dump & joblib.load is more efficient than pickle on objects with large numpy arrays
# internally as is often the case for fitted scikit-learn estimators, (can only pickle to the disk and not to a string)

import sys, os
sys.path.append('../')
from DigitClassifier import DigitClassifier

UPLOAD_FOLDER = './uploads/'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# @app.route('/', methods=['GET', 'POST'])
# def upload_file():
#     if request.method == 'POST':
#         # check if the post request has the file part
#         if 'file' not in request.files:
#             flash('No file part')
#             return redirect(request.url)
#         file = request.files['file']
#         # if user does not select file, browser also
#         # submit a empty part without filename
#         if file.filename == '':
#             flash('No selected file')
#             return redirect(request.url)
#         if file and allowed_file(file.filename):
#             filename = secure_filename(file.filename)
#             file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#             return redirect(url_for('uploaded_file',
#                                     filename=filename))
    # return '''
    # <!doctype html>
    # <title>Upload new File</title>
    # <h1>Upload new File</h1>
    # <form method=post enctype=multipart/form-data>
    #   <p><input type=file name=file>
    #      <input type=submit value=Upload>
    # </form>
    # '''

@app.route('/upload', methods=['POST'])#'GET', 'POST'])
def upload():
	# if request.method=='GET':
	# 	#return('<form action="/upload" method="post"><input type="submit" value="Upload" /></form>')
	# 	return '''
	#     	<!doctype html>
	#     	<html>
	#     	<body>
	#     	<form action='/upload' method='post' enctype='multipart/form-data'>
	#       		<input type='file' name='file'>
	#         	<input type='submit' value='Upload'>
	#     	</form>
	#     	'''

	if request.method=='POST':
		input_file = request.files['file'] # werkzeug.datastructures.FileStorage instance

		# if user does not select file, submit a empty part without filename
		if input_file.filename == '': # file name (without path)
			print "no selected file! ", input_file, input_file.filename #flash('No selected file')
			return redirect(request.url)
		elif input_file and allowed_file(input_file.filename):
			print "allowed ",input_file.filename
			# filename = secure_filename(input_file.filename)
			# input_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			#return redirect(url_for('uploaded_file', filename=filename))
			print "Classifying image... ",input_file.filename,type(input_file.filename)
			dc = DigitClassifier("CNN")
			# Load trained model
			model = dc.load_model()
			model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

			# evaluate loaded model on test data
			#filename = 'data/test/4391.png'
			predicted_label = dc.predict_image(input_file, model)
			print "Predictions with loaded model for image: ", predicted_label
			return 'Predicted number for input image: %s ' % (predicted_label)
		else:
			return "file input format not allowed or was empty"
	else:
		return "ok" # a function or a string must be return



@app.route('/')
def index():
	return '''
    	<!doctype html>
    	<html>
    	<body>
    	<form action='/upload' method='post' enctype='multipart/form-data'>
      		<input type='file' name='file'>
        	<input type='submit' value='Upload'>
    	</form>
    	'''

if __name__ == '__main__':
	"""
    In order to not restart the local server after each change to your code, if you enable debug
    support the server will reload itself on code changes, and provide a debugger. To enable it,
    before running the server:
    $ export FLASK_DEBUG=1
    $ flask run
    or, app.run(threaded=True) allows stopping the server with ctr-c
    """
	# with app.test_request_context():
	# 	print url_for('index')
    #     print url_for('login')
    #     print url_for('login', next='/')
    #     print url_for('profile', username='John Doe')
    #     print url_for('static', filename='style.css')
	app.debug = True
	app.run(threaded=True, debug=True)#, host="0.0.0.0") # Default is: threaded=True allows stopping the server with ctr-c
