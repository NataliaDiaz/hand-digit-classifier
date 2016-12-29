import httplib


image_path = "./data/test/3.png"
conn = httplib.HTTPConnection("localhost:8009")
conn.request("POST", "/mnist/classify", image_path)
#conn.send(image_path)
response = conn.getresponse()
conn.close()

print "Client Response after sending clientdata: ", image_path,": ", response.read()
