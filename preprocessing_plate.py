import cv2
import numpy as np
from PIL import Image
import easyocr
from RealESRGAN import RealESRGAN
import Levenshtein

def find_most_similar_string(query, database):
    most_similar = None
    max_similarity = 0

    for entry in database:
        # Levenshtein distance 계산
        similarity = Levenshtein.ratio(query, entry)
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar = entry

    return most_similar, max_similarity

def preprocessing_plate(license_plate_detection, device, model_weights_path='weights/RealESRGAN_x4.pth'):
 
    database = ["91로4775", "98도7265", "경남06도3190", "경남06모6438", "경남06모7443", "경남06모7848", "경남06보5360", 
                "경남06소6818", "경남06소6819", "경남82사4334", "경남99사4612", "경북06모6948", "경북06모7391", "경북82아2891", 
                "부산06마5777", "부산94아2764", "부산94아3650"]

    model_RealESRGAN = RealESRGAN(device, scale=4)  
    model_RealESRGAN.load_weights(model_weights_path, download=True)


        # RealESRGAN + CLAHE 적용
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))  # CLAHE 설정
    gray_img_np = cv2.cvtColor(license_plate_detection, cv2.COLOR_RGB2GRAY)  # 그레이스케일로 변환
    clahe_img_np = clahe.apply(gray_img_np)  # CLAHE 적용
    img_pil = Image.fromarray(clahe_img_np)  # numpy -> PIL
    img_pil = img_pil.convert('RGB')  # RGB로 변환
    img_pil = model_RealESRGAN.predict(img_pil)  # RealESRGAN으로 해상도 개선
    img_np = np.array(img_pil)  # PIL -> numpy로 변환

    gray_img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)  # 그레이스케일 변환

        # Non-Local Means (NLM) 필터로 노이즈 제거
    denoised_img_np = cv2.fastNlMeansDenoising(gray_img_np, None, 30, 7, 21)

        # Canny 엣지 검출
    edges_np = cv2.Canny(denoised_img_np, 100, 200)

        # 허프 직선 변환을 이용해 직선 검출
    lines_np = cv2.HoughLinesP(edges_np, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=10)

        # 직선의 각도 계산
    angles = []
    for line in lines_np:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))  # 기울기 계산
        angles.append(angle)

        # 각도가 너무 분산되거나 직선이 적으면 0 각도를 사용
    if len(angles) > 0:
        median_angle = np.median(angles)
    else:
        median_angle = 0  # 직선이 없다면 각도 0으로 처리
        
        # 이미지 크기와 중심 계산
    (h, w) = gray_img_np.shape[:2]
    center = (w // 2, h // 2)

        # 회전 행렬 생성
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)

        # 이미지를 회전시켜 기울기 보정
    rotated_img_np = cv2.warpAffine(img_np, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)

        # 결과 이미지 PIL 객체로 변환
    rotated_img_pil = Image.fromarray(rotated_img_np)

        # 대비 및 밝기 조정
    alpha = 1.5  # 대비 강도
    beta = -100   # 밝기 조정
    adjusted_image = cv2.convertScaleAbs(rotated_img_np, alpha=alpha, beta=beta)

        # 이미지 샤프닝
    sharpen_kernel = np.array([[-1, -1, -1],
                                [-1, 9, -1],
                                [-1, -1, -1]])
    sharpened_image = cv2.filter2D(adjusted_image, -1, sharpen_kernel)

        # EasyOCR로 텍스트 인식
    reader = easyocr.Reader(['ko'], gpu=True)
    resultss = reader.readtext(sharpened_image)

        # 번호판 주변에 박스를 그리기 및 텍스트 표시
    # for bbox, text, _ in resultss:
    #     pts = np.array(bbox, dtype=np.int32)
    #     pts = pts.reshape((-1, 1, 2))
    #     sharpened_image = cv2.polylines(sharpened_image, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
    #     sharpened_image = cv2.putText(sharpened_image, text, (pts[0][0][0], pts[0][0][1] - 10),
    #                                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

        # 한글과 숫자만 추출 (기호 제외)
    filtered_text = []
    for (_, text, _) in resultss:
        filtered = ''.join([char for char in text if char.isdigit() or ('가' <= char <= '힣')])
        if filtered:  # 빈 문자열을 추가하지 않도록
            filtered_text.append(filtered)

    result =  ''.join(filtered_text)

    most_similar, similarity = find_most_similar_string(result, database)
    if most_similar is None:
        most_similar = "알 수 없음"  # 적절한 기본값으로 설정


    return sharpened_image, most_similar
