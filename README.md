# hand-digit-classifier

MNIST Handwritten Digit Classifier flask application   `diaz.rodriguez.natalia@gmail.com`


How to run the flask application's builtin server (runs on http://127.0.0.1:5000/):

```$ export FLASK_APP=digit_recognition.py
$ flask run
```

Alternatively you can use (to run on http://127.0.0.1:5000/):

```
$ export FLASK_APP=digit_recognition.py
$ python -m flask run
```

Now head over to http://127.0.0.1:5000/, and you should see the application where an image can be uploaded for recognition. In order to download test and train images, run:

```python
dc = DigitClassifier()
dc.download_imgs("data/train.csv", "data/test.csv", "./data/train/", "./data/test/")
```

To train and predict with DigitClassifier:

```python
model = dc.train()
#predicted = dc.predict(model, "/data/predicted/")
dc.evaluate(model)
```


Note: to make the server externally visible, if you have the debugger disabled or trust the users on your network, you can make the server publicly available simply by adding --host=0.0.0.0 to the command line:

```
flask run --host=0.0.0.0
```

This tells your operating system to listen on all public IPs. If the server does not start (or to log errors and stack traces in Debug mode), before running the server:

```
$ export FLASK_DEBUG=1
$ flask run
```






# Requirements:
1. Create and activate keras virtual environment in Python:

```
virtualenv kerasenv
```

within the env:

```
source kerasenv/bin/activate
```


2. Option a: Install keras docker, nvidia drivers and nvidia docker file.

2. Option b: Install
```
pip install numpy jupyter keras matplotlib
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
