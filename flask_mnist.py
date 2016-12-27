# -*- coding:utf-8 -*-
from flask import Flask, url_for
from flask import render_template
from flask import abort, redirect, url_for
from flask import request

##
from scipy import misc
from sklearn.externals import joblib

app = Flask(__name__)

"""
This program creates a service using a python web framework in Flask that takes
an MNIST (*) image as a post request and returns a JSON blob with the classification
of that image. 
(*) http://yann.lecun.com/exdb/mnist/

```
POST /mnist/classify
    Returns the class of the image. Invalid input should return a 404.
```

Instructions: to launch the builtin server:
export FLASK_APP=hello.py
$ python -m flask run
 * Running on http://127.0.0.1:5000/
Now head over to http://127.0.0.1:5000/, and you should see your hello world greeting.
"""

@app.errorhandler(404)
def not_found(error):
    return render_template('error.html'), 404
    # resp = make_response(render_template('error.html'), 404)
    # resp.headers['X-Something'] = 'A value'
    # return resp

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['the_file']
        f.save('/var/www/uploads/uploaded_file.txt')
        app.logger.debug('file uploaded')
        #app.logger.warning('A warning occurred (%d apples)', 42)
        #app.logger.error('An error occurred')

@app.route('/')
def index():
    return 'Index Page'
    # return '''
    # <!doctype html>
    # <html>
    # <body>
    # <form action='/upload' method='post' enctype='multipart/form-data'>
    #     <input type='file' name='file'>
    # <input type='submit' value='Upload'>
    # </form>
    # '''
    # To redirect a user to another endpoint:
    #return redirect(url_for('login'))

# @app.route('/hello')
# def hello():
#     return 'Hello, World'

@app.route('/hello/')
@app.route('/hello/<name>')
def hello(name=None):
    return render_template('hello.html', name=name)


@app.route('/user/<username>')
def show_user_profile(username):
    # show the user profile for that user
    return 'User %s' % username

@app.route('/post/<int:post_id>')
def show_post(post_id):
    # show the post with the given id, the id is an integer
    return 'Post %d' % post_id

@app.route('/mnist/')
def projects():
    return 'The mnist flask project page'

@app.route('/about')
def about():
    return 'The about page: Natalia Diaz Rodriguez, Dec 2016 diaz.rodriguez.natalia@gmail.com'

@app.route('/user/<username>')
def profile(username): pass

# @app.route('/login')
# def login():
#     # to abort a request early with an error code, use the abort() function:
#     abort(401)
#     this_is_never_executed()

@app.errorhandler(404)
def page_not_found(error):
    return render_template('page_not_found.html'), 404

def load_model(path):
    from sklearn.externals import joblib
    return joblib.load('path')

@app.route('/login', methods=['POST', 'GET'])
def login():
    error = None
    if request.method == 'POST':
        # do the login
        if valid_login(request.form['username'],
                       request.form['password']):
            return log_the_user_in(request.form['username'])
        else:
            error = 'Invalid username/password'
    else:
        print show_login_form()
    # the code below is executed if the request method
    # was GET or the credentials were invalid
    return render_template('login.html', error=error)

def CNN_predict(input_img):
    return output_img

with app.test_request_context('/hello', method='POST'):
    # now you can do something with the request until the
    # end of the with block, such as basic assertions:
    assert request.path == '/hello'
    assert request.method == 'POST'

@app.route('/mnist', methods=['POST'])
def mnist():
    input = ((255 - np.array(request.json, dtype=np.uint8)) / 255.0).reshape(1, 784)
    output = CNN_predict(input)
    return jsonify(results=[output])

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    f = request.files['file']
    im = misc.imread(f)
    img = im.reshape((1,784))
    CNN_predict = joblib.load('model/ok.m')
    l = CNN_predict(img)
    return 'predict: %s ' % (l[0])

if __name__ == '__main__':
    """
    The flask script is nice to start a local development server, but you would have to restart it
    manually after each change to your code. To enable debug support so that the server will reload itself on code changes, before running the server:
    $ export FLASK_DEBUG=1
    $ flask run
    or, app.run(threaded=True) allows stopping the server with ctr-c
    """
    with app.test_request_context():
        print url_for('index')
        print url_for('login')
        print url_for('login', next='/')
        print url_for('profile', username='John Doe')
        print url_for('static', filename='style.css')

    app.run(threaded=True)# host="0.0.0.0") # Default is: threaded=True allows stopping the server with ctr-c