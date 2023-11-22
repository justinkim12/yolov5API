import torch
import cv2

import numpy as np
import json

from PIL import Image, ImageDraw
from ultralytics.utils.plotting import Annotator, colors

# model 가져오기 / 경로에 pt 파일이 있어야함
foo = torch.hub.load("ultralytics/yolov5", 'custom', 'best.pt', force_reload=True, skip_validation=True)

# img 파일을 읽어왔다고 가정
# -> local에서 2장 읽어온 후 list에 저장
img = []
for i in range(1, 3):
    image = Image.open(f"IMG{i}.jpg").convert('RGB')
    img.append(image)

# model이 추론 진행 후 결과 반환
result = foo(img, size=640)

# 사진에 bounding box 그리고, class 별 객체 수 세서 json 형식으로 변환
names = result.names
class_dict = dict()
dis_type = ['scratch', 'installation', 'exterior', 'gap']
for type in dis_type:
    class_dict[type] = 0

for i, det in enumerate(result.pred):
    x = np.ascontiguousarray(img[i])
    annotator = Annotator(x, line_width=3, example=str(names))
    for *xyxy, conf, cls, in reversed(det):
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
print(output)


def getBoundingBoxes(img):
    for i in range(len(img)):
        draw = ImageDraw.Draw(img[i])

        # bounding box 그리기
        for j in range(len(result.pandas().xyxy[i].xmin)):
            xmin = result.pandas().xyxy[i].xmin[j]
            xmax = result.pandas().xyxy[i].xmax[j]
            ymin = result.pandas().xyxy[i].ymin[j]
            ymax = result.pandas().xyxy[i].ymax[j]

            draw.rectangle((xmin, ymin, xmax, ymax), outline=(255, 0, 0), width=5)

        # img[i].show()
        # img[i].save(f'./photo/output_image_{i}.png')
    return img
