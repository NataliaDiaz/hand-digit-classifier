from BaseHTTPServer import BaseHTTPRequestHandler,HTTPServer
import os
#from DigitClassifier import DigitClassifier

ADDR = "localhost"
PORT = 8009

# SERVER
class RequestHandler(BaseHTTPRequestHandler):
    #def __init__(self):

    def do_POST(self):
        #digit_classifier = DigitClassifier()
        classification_label = [-1]
        print "Path: ",self.path

        length = int(self.headers['Content-length'])
        input_image = self.rfile.read(length)
        print "Content: ", input_image
        input_folder = "./data/test/"

        if isinstance(input_image, str) and input_image.startswith(input_folder):
            #classification_label = digit_classifier.predict(input_image)[0]
            self.send_response(200, ("OK! classification result is "+str(classification_label)))
            print "Classification result: ", str(classification_label[0])
        else:
            print "Error: the input image needs to be a file name (string) pre-processed as in "+input_folder
            self.send_response(404, "Error: the input image needs to be a file name (string) pre-processed as in "+input_folder)
        self.end_headers()
        self.wfile.write(classification_label)
        return classification_label #jsonify(classification_label)

httpd = HTTPServer((ADDR, PORT), RequestHandler)
httpd.serve_forever()
