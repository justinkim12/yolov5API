# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run a Flask REST API exposing one or more YOLOv5s models
"""

import argparse
import base64
import io
import json
import numpy as np
from ultralytics.utils.plotting import Annotator, colors

import torch
from flask import Flask, request, jsonify, send_file
from PIL import Image, ImageDraw

# from test import getBoundingBoxes

app = Flask(__name__)

DETECTION_URL = '/object-detection/best'



def getBoundingBoxes(img,result):
    for i in range (len(img)) :
        draw = ImageDraw.Draw(img[i])

        # bounding box 그리기
        for j in range (len(result.pandas().xyxy[i].xmin)) :
            xmin = result.pandas().xyxy[i].xmin[j]
            xmax = result.pandas().xyxy[i].xmax[j]
            ymin = result.pandas().xyxy[i].ymin[j]
            ymax = result.pandas().xyxy[i].ymax[j]

            draw.rectangle((xmin, ymin, xmax, ymax), outline=(255,0,0), width = 5)

        # img[i].show()
        # img[i].save(f'./photo/output_image_{i}.png')
    return img

@app.route('/v2' + DETECTION_URL, methods=['POST'])
def getFile():
    if request.method != 'POST':
        return
    # 다중 파일 업로드 처리
    imgs = []
    if request.files.getlist('images'):
        for i, file in enumerate(request.files.getlist('images')):
            im_file = file
            im_bytes = im_file.read()
            im = Image.open(io.BytesIO(im_bytes))  # TODO2 이미지 처리 이거 맞나?
            imgs.append(im)

        result = model(imgs, size=640)

        names = result.names
        class_dict = dict()
        for i, det in enumerate(result.pred):
                x = np.ascontiguousarray(imgs[i])
                annotator = Annotator(x, line_width=3, example=str(names))
                for *xyxy, cls, in reversed(det):
                    c = int(cls)
                    label = f'{names[c]}'

                    key = names[c]
                    if (key in class_dict):
                        value = class_dict.get(key) + 1
                        class_dict.update({key: value})
                    else:
                        class_dict[key] = 1

                    annotator.box_label(xyxy, label, color=colors(c, True))
                # bounding box 그려서 img에 저장
                imgs[i] = Image.fromarray(annotator.result())
        # json 파일로 변환
        data = class_dict


        img = getBoundingBoxes(imgs,result)

        encoded_imges = []
        for i, img in enumerate(imgs):
            # img.show()
            buffer = io.BytesIO()
            img.save(buffer,format='png')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            encoded_imges.append(image_base64)
        return jsonify({'result': data,'images':encoded_imges})

    return json.dumps({"result": "fail"})


@app.route('/v1/' + DETECTION_URL, methods=['POST'])
def predict():
    if request.method != 'POST':
        return
    # 다중 파일 업로드 처리
    if request.files:
        # Method 1
        # with request.files["image"] as f:
        #     im = Image.open(io.BytesIO(f.read()))

        # Method 2
        files = request.files.getlist('files[]')

        im_file = files[0]
        im_bytes = im_file.read()
        im = Image.open(io.BytesIO(im_bytes))

        img = []
        img.append(im)
        '''
        이미지 읽어 오는 부분에서 한 장씩 처리 or 여러 장 한 번에 처리
        우선 전체적으로 여러 장을 리스트에 넣어서 한 번에 처리하는 방식으로 구현을 했습니다!
        '''

        result = model(img, size=640)

        names = result.names
        class_dict = dict()
        for i, det in enumerate(result.pred):
            x = np.ascontiguousarray(img[i])
            annotator = Annotator(x, line_width=3, example=str(names))
            for *xyxy, cls, in reversed(det):
                c = int(cls)
                label = f'{names[c]}'

                key = names[c]
                if (key in class_dict):
                    value = class_dict.get(key) + 1
                    class_dict.update({key: value})
                else:
                    class_dict[key] = 1

                annotator.box_label(xyxy, label, color=colors(c, True))
            # bounding box 그려서 img에 저장
            img[i] = Image.fromarray(annotator.result())
        # json 파일로 변환
        output = json.dumps(class_dict)

        return output


if __name__ == '__main__':
    default_port = 5000
    model = torch.hub.load("ultralytics/yolov5", 'custom', 'best.pt', force_reload=True, skip_validation=True)

    app.run(host='0.0.0.0', port=default_port)  # debug=True causes Restarting with stat
    app.logger.info("Hi")
