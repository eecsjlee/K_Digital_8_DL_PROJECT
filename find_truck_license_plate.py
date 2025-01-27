import torch
import yolov5
import easyocr

def find_truck_license_plate(img, device):
     model_license_plate = yolov5.load('keremberke/yolov5m-license-plate')
     model_license_plate.conf = 0.25  # 신뢰도 기준 25%이상
     model_license_plate.iou = 0.45  # 박스 겹침 정도, 45%이상 겹치는 박스 중복탐지로 제거
     model_license_plate.agnostic = False  # 객체 종류별로 중복을 처리
     model_license_plate.multi_label = False  # 하나의 객체에 여러 라벨 붙는 경우를 방지
     model_license_plate.max_det = 1000  # 한이미지당 최대 탐지 객체 수 1000개
     model_license_plate.to(device)
     reader = easyocr.Reader(['ko'], gpu=True)

     
     result = model_license_plate(img, size=640)  # 이미지 리사이즈 640 * 640
     result = model_license_plate(img, augment=True) #데이터 증강 활성화, 회전,확대등
     license_plate_detections = None
     results = []
     result_img = result.ims[0]
     result_xywh = torch.tensor(result.xywh[0])
     license_plate_class_id = 0  # 번호판 클래스 ID (모델에 따라 번호판 클래스가 다를 수 있음)

     filtered_indices = result_xywh[result_xywh[:, 5] == license_plate_class_id]

     for detection in filtered_indices:
          x_center, y_center, width, height, confidence, class_id = detection
          x_min = int((x_center - width / 2))  # x1 좌표
          y_min = int((y_center - height / 2))  # y1 좌표
          x_max = int((x_center + width / 2))  # x2 좌표
          y_max = int((y_center + height / 2))  # y2 좌표
          # 번호판 영역만 추출
          license_plate_roi = result_img[y_min:y_max, x_min:x_max]
          results = reader.readtext(license_plate_roi)
          if results :
               license_plate_detections = license_plate_roi  # 텍스트를 찾은 경우
               break
    
     return license_plate_detections, results