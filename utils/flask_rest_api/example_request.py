# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Perform test request
"""

import pprint

import requests
import json
DETECTION_URL = 'http://127.0.0.1:5000/v2/object-detection/best'
IMAGES = ['IMG1.jpg','IMG2.jpg']

# Read image
dict=[]
for i,IMAGE in enumerate(IMAGES):
    with open(IMAGE, 'rb') as f:
        image_data=f.read()
        dict.append(image_data)

response = requests.post(DETECTION_URL, files={'images': dict[0],'images': dict[1]}).json()

pprint.pprint(response)
