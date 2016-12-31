# -*- coding:utf-8 -*-
from flask import Flask, request
from flask import redirect, url_for
from werkzeug.utils import secure_filename
from sklearn.externals import joblib
# joblib.dump & joblib.load is more efficient than pickle on objects with large numpy arrays
# internally as is often the case for fitted scikit-learn estimators, (can only pickle to the disk and not to a string)

import sys, os
sys.path.append('../')
from DigitClassifier import DigitClassifier#, create_path_if_doesnt_exist

UPLOAD_FOLDER = './uploads/'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
#app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/mnist/classify', methods=['POST'])#'GET', 'POST'])
def classify():
	if request.method=='POST':
		input_file = request.files['file'] # werkzeug.datastructures.FileStorage instance

		# if user does not select file, submit a empty part without filename
		if input_file.filename == '': # file name (without path)
			print "no selected file! ", input_file, input_file.filename #flash('No selected file')
			return redirect(request.url)
		elif input_file and allowed_file(input_file.filename):
			#filename = secure_filename(input_file.filename)
            #create_path_if_doesnt_exist(UPLOAD_FOLDER)
			#input_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			dc = DigitClassifier("CNN")
			# Load trained model
			model = dc.load_model()
			model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

			# evaluate loaded model on test data
			#filename = 'data/test/4391.png'
			predicted_label = dc.predict_image(input_file, model)
			print "Predictions with loaded model for image: ",input_file.filename, type(input_file.filename),": ", predicted_label
			return 'Predicted number for input image: %s ' % (predicted_label)
            #return redirect(url_for('uploaded_file', filename=filename))
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
    	<form action='/mnist/classify' method='post' enctype='multipart/form-data'>
      		<input type='file' name='file'>
        	<input type='submit' value='Upload'>
    	</form>
    	'''

@app.errorhandler(404)
def page_not_found(error):
    return render_template('page_not_found.html'), 404


if __name__ == '__main__':
	"""
    In order to not restart the local server after each change to your code, if you enable debug
    support the server will reload itself on code changes, and provide a debugger. To enable it,
    before running the server:
    $ export FLASK_DEBUG=1
    $ flask run
    or, app.run(threaded=True) allows stopping the server with ctr-c
    """
	app.run(threaded=True, debug=True)#, host="0.0.0.0") # Default is: threaded=True allows stopping the server with ctr-c
