import cv2
import easyocr

def find_license_plate_boxes(license_plate):
    # 1. 이미지를 그레이스케일로 변환
    gray = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)
    
    # 2. 이미지 전처리 (이진화)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 3. 윤곽선 검출
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 4. 윤곽선에서 최소 사각형 박스 추출
    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # 필터링 조건 (너무 작은 박스 제거)
        if w > 5 and h > 10:  # 너비와 높이에 대한 필터링을 통해 작은 영역 제거
            bounding_boxes.append((x, y, w, h))
    
    # 5. 박스를 좌측에서 우측 순으로 정렬
    bounding_boxes = sorted(bounding_boxes, key=lambda box: box[0])

    img_copy = license_plate.copy()  # 이미지 복사본 생성
    
    # 6. 이미지에 사각형 박스를 그려서 결과 반환
    for box in bounding_boxes:
        x, y, w, h = box
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    return bounding_boxes, img_copy

# def find_license_plate_boxes(license_plate_image):
#     # 1. 이미지를 그레이스케일로 변환
#     gray = cv2.cvtColor(license_plate_image, cv2.COLOR_BGR2GRAY)
    
#     # 2. 이미지 전처리 (이진화)
#     _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
#     # 3. EasyOCR 초기화 (한글, 영어 인식)
#     reader = easyocr.Reader(['ko', 'en'])  # 한국어와 영어를 모두 인식
    
#     # 4. 번호판 영역을 OCR로 처리 (이미지에서 텍스트 추출)
#     results = reader.readtext(binary)
    
#     # 5. 결과 출력
#     license_plate_text = []
#     bounding_boxes = []
    
#     img_copy = license_plate_image.copy()  # 이미지 복사본 생성
    
#     # 6. OCR로 인식된 텍스트에 박스 그리기
#     for result in results:
#         text = result[1]  # OCR 인식된 텍스트
#         print(f"Detected text: {text}")
        
#         # 텍스트의 좌표로 박스를 그리기
#         points = result[0]  # 4개의 좌표점 [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
        
#         # 각 점에서 최소, 최대 x, y 값을 추출
#         x_min = min([point[0] for point in points])
#         y_min = min([point[1] for point in points])
#         x_max = max([point[0] for point in points])
#         y_max = max([point[1] for point in points])
        
#         # 각 좌표값을 정수로 변환 (cv2.rectangle()에 필요한 형식)
#         x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
        
#         # 숫자 및 한글 인식 시 박스를 그려서 표시
#         if any(c.isdigit() for c in text) or any('\u3130' <= c <= '\u316f' or '\uac00' <= c <= '\ud7a3' for c in text):
#             cv2.rectangle(img_copy, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # 텍스트가 숫자나 한글이면 박스 그리기
#             license_plate_text.append(text)  # 텍스트 저장
#             bounding_boxes.append((x_min, y_min, x_max - x_min, y_max - y_min))  # 박스 좌표 저장
    
#     return license_plate_text, img_copy, bounding_boxes
