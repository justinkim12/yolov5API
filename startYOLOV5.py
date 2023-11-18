import torch
import os
import argparse
import io

from flask import Flask, request
from PIL import Image
import sys
sys.path.insert(0, '')

os.environ['KMP_DUPLICATE_LIB_OK']='True'


"""
Run a Flask REST API exposing one or more YOLOv5s models
"""


app = Flask(__name__)
models = {}

DETECTION_URL = '/v1/object-detection'


@app.route(DETECTION_URL, methods=['POST'])
def predict():
    # logger= __get_logger() #TODO1 로그는 대체 어떻게 찍는가 

    if request.method != 'POST':
        return

    if request.files.get('image'):

        im_file = request.files['image']
        im_bytes = im_file.read()
        im = Image.open(io.BytesIO(im_bytes)) #TODO2 이미지 처리 이거 맞나?

        results = model(im, size=640)  #TODO3 우리의 모델의 input과 output이 어떻게 나올까?
        return results.pandas().xyxy[0].to_json(orient='records')
    return "No Image!"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Flask API exposing YOLOv5 model')
    parser.add_argument('--port', default=5000, type=int, help='port number')
    parser.add_argument('--model', nargs='+', default=['yolov5s'], help='model(s) to run, i.e. --model yolov5n yolov5s')
    opt = parser.parse_args()
    model = torch.load('test.pt')#TODO3 질문 우리의 모델의 위치는 어디인가
    # model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True)
    app.run(host='0.0.0.0', port=opt.port)  # debug=True causes Restarting
    # with stat



#모델이 얼마나 무거워요?
#경량화
#30GB 1GB 1순위 AWS
#2순위 학교서버
#3순위 개발 컴퓨터에서 포트 포워딩