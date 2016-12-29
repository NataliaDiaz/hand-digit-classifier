# hand-digit-classifier

MNIST Handwritten Digit Classifier flask application 



How to run the flask application's builtin server (runs on http://127.0.0.1:5000/):


```
. venv/bin/activate
$ export FLASK_APP=digit_recognition_webapp.py
$ THEANO_FLAGS='blas.ldflags='   flask run
```

Alternatively you can use:

```
$ THEANO_FLAGS='blas.ldflags=' python -m flask run
```

Head over to http://127.0.0.1:5000/, and you should see the application where an image can be uploaded for recognition.

Note: to make the server externally visible, if you have the debugger disabled or trust the users on your network, you can make the server publicly available simply by adding --host=0.0.0.0 to the command line:

```
THEANO_FLAGS='blas.ldflags=' flask run --host=0.0.0.0
```

This tells your operating system to listen on all public IPs. If the server does not start (or to log errors and stack traces in Debug mode), before running the server:

```
$ export FLASK_DEBUG=1
$ THEANO_FLAGS='blas.ldflags=' flask run
```


# Model usage


To run DigitClassifier and download test and train images, see `DigitClassifier.py`. It can be run as `THEANO_FLAGS='blas.ldflags=' python DigitClassifier.py`  (Flag is needed in the command line to prevent error in [1]).

```python
from DigitClassifier import DigitClassifier
dc = DigitClassifier()
dc.download_imgs("data/train.csv", "data/test.csv", "./data/train/", "./data/test/")
```
To train and predict with DigitClassifier:

```python
model = dc.train()
predicted = dc.predict(model)
dc.evaluate(model)
```



# Requirements:
1. Create and activate keras virtual environment in Python:

```
virtualenv venv
```



and within the env folder, type: `source venv/bin/activate`


2. Option a: Install keras docker, nvidia drivers and nvidia docker file. Theano for the backend.

2. Option b: Install
```
pip install numpy jupyter keras matplotlib h5py
```


---


[1]


```
Error in error.txt : File "DigitClassifier.py", line 151, in train
validation_data=(self.X_test, self.Y_test))
File "/Users/natalia/planet/flask_mnist/venv/lib/python2.7/site-packages/Keras-1.2.0-py2.7.egg/keras/models.py", line 671, in fit
initial_epoch=initial_epoch)
File "/Users/natalia/planet/flask_mnist/venv/lib/python2.7/site-packages/Keras-1.2.0-py2.7.egg/keras/engine/training.py", line 1085, in fit
self._make_test_function()
...
File "/Users/natalia/planet/flask_mnist/venv/lib/python2.7/site-packages/Theano-0.8.2-py2.7.egg/theano/gof/cmodule.py", line 2204, in compile_str
(status, compile_stderr.replace('\n', '. ')))
Exception: ('The following error happened while compiling the node', Dot22(dense_input_1, dense_1_W), '\n', 'Compilation failed (return status=1): ld: library not found for -lopenblas. clang: error: linker command failed with exit code 1 (use -v to see invocation). ', '[Dot22(dense_input_1, dense_1_W)]')
```


# Dataset:
[MNIST](http://yann.lecun.com/exdb/mnist/) TRAINING SET LABEL FILE (train-labels-idx1-ubyte):

```
[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000801(2049) magic number (MSB first)
0004     32 bit integer  60000            number of items
0008     unsigned byte   ??               label
0009     unsigned byte   ??               label
........
xxxx     unsigned byte   ??               label
```

The labels values are 0 to 9.

All digit images have been size-normalized and centered in a fixed size image of 28 x 28 pixels. Pixels are organized row-wise. In the original dataset each pixel of the image is represented by a value between 0 and 255, where 0 is black, 255 is white and anything in between is a different shade of grey.


# Author

Natalia Diaz Rodriguez, ```diaz.rodriguez.natalia@gmail.com```
