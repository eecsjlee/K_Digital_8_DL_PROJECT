import numpy as np
import random
from trdg.generators import GeneratorFromStrings
import os
from tqdm.auto import tqdm
from PIL import Image, ImageColor, ImageFilter
import random
import cv2

# --- 설정값 정의 ---
# 생성할 이미지 개수
NUM_IMAGES_TO_SAVE = 1500

# 사용할 한글 리스트 (일부 글자만 사용)
KOREAN_CHARS = ["가", "거", "고", "구", "나", "너", "노", "누", "다", "더", "도", "두", "라", "러", "로", "루", "마", "머", "모", "무", "버", "보", "부", "서", "소", "수", "어", "오", "우", "저", "조", "주", "하", "허", "호"]
# 사용할 지역 한글 리스트 (예시)
REGION_KOREAN_CHARS = ["서울", "경기", "인천", "강원", "충남", "충북", "대전", "경북", "경남", "대구", "울산", "부산", "광주", "전남", "전북", "제주"]

# 출력 폴더 설정
OUTPUT_DIR = 'output'

# --- 번호판 문자열 생성 함수 ---
def generate_number_strings(num_images):
    """
    다양한 형식의 번호판 문자열을 생성합니다.
    """
    number_strings = []

    # 70%는 지역 번호판 (지역 한글 2자리 + 숫자 2~3자리 + 한글 1자리 + 숫자 4자리)
    for _ in range(num_images * 7 // 10):
        region_char = random.choice(REGION_KOREAN_CHARS)
        front_digits = np.random.randint(10, 1000)
        if random.random() < 0.5:
            front_digits = np.random.randint(10, 100)
        korean_char = random.choice(KOREAN_CHARS)
        back_digits = np.random.randint(1000, 10000)
        number_strings.append(f"{region_char}{front_digits}{korean_char}{back_digits}")

    # 20%는 신형 번호판 (숫자 3자리 + 한글 1자리 + 숫자 4자리)
    for _ in range(num_images * 2 // 10):
        front_digits = np.random.randint(100, 1000)
        korean_char = random.choice(KOREAN_CHARS)
        back_digits = np.random.randint(1000, 10000)
        number_strings.append(f"{front_digits}{korean_char}{back_digits}")

    # 10%는 구형 번호판 (숫자 2자리 + 한글 1자리 + 숫자 4자리)
    for _ in range(num_images * 1 // 10):
        front_digits = np.random.randint(10, 100)
        korean_char = random.choice(KOREAN_CHARS)
        back_digits = np.random.randint(1000, 10000)
        number_strings.append(f"{front_digits}{korean_char}{back_digits}")

    return number_strings

def get_random_text_color():
    """검정 또는 흰색 또는 회색을 랜덤으로 선택"""
    return random.choice(["#000000", "#ffffff", "#808080"])


def add_gaussian_noise(img, mean=0, std=None):
    """이미지에 Gaussian noise 추가"""
    if std is None:
        std = random.uniform(10, 30)
    img_np = np.array(img)
    noise = np.random.normal(mean, std, img_np.shape).astype(np.uint8)
    noisy_img = np.clip(img_np + noise, 0, 255).astype(np.uint8)  # 값을 0 ~ 255 사이로 유지
    return Image.fromarray(noisy_img)

# --- 이미지 생성 및 저장 함수 ---
def generate_and_save_images(output_dir, num_images, strings):
    """
    주어진 문자열 목록을 사용하여 이미지를 생성하고, 출력 폴더에 저장합니다.
    """
    # 출력 폴더 생성 (이미 있다면 건너뜀)
    os.makedirs(output_dir, exist_ok=True)

    # 라벨 파일 생성 (이미 있다면 덮어씀)
    labels_path = os.path.join(output_dir, "labels.txt")
    with open(labels_path, "w", encoding='utf-8') as f:
        pass  # 파일 생성 후 닫음

    # 이미지 생성 및 저장
    current_index = 0
    with open(labels_path, "a", encoding='utf-8') as f:
        for counter, lbl in tqdm(enumerate(strings), total=num_images, desc="Generating images"):
             if counter >= num_images:
                 break
             text_color = get_random_text_color()
            # 이미지 생성기 생성 (배경색 없음)
             generator_with_color = GeneratorFromStrings(
                 strings=[lbl],  # 글자색 변경을 위해 단일 글자 생성기 사용
                 count=1,
                 language='ko',
                 text_color=text_color,
                 blur = random.uniform(0.5, 1.5)
              )
             img, _ = next(generator_with_color)  # 이미지 생성

             # 배경색 설정 (진한 회색)
             if text_color == "#000000":
                background_color = ImageColor.getrgb("#404040")
                new_img = Image.new("RGB", img.size, background_color)
                new_img.paste(img, mask=img.convert("RGBA").split()[3])
                img = new_img
             elif text_color == "#ffffff":
                background_color = ImageColor.getrgb("#101010")
                new_img = Image.new("RGB", img.size, background_color)
                new_img.paste(img, mask=img.convert("RGBA").split()[3])
                img = new_img
             elif text_color == "#808080":
                background_color = ImageColor.getrgb("#101010")
                new_img = Image.new("RGB", img.size, background_color)
                new_img.paste(img, mask=img.convert("RGBA").split()[3])
                img = new_img


             img = add_gaussian_noise(img, std=random.uniform(1, 5))
             img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))  # 가우시안 블러 추가

             # 이미지 회전
             random_angle = random.uniform(-10, 10)
             img = img.rotate(random_angle, fillcolor=background_color if text_color != "#ffffff" else (0,0,0)) # 회전 적용

            # 이미지 객체가 제대로 생성되었는지 확인
             if not isinstance(img, Image.Image):
                print(f"Error: Invalid image object received at index {current_index}. Type: {type(img)}")
                continue

             img_path = os.path.join(output_dir, f'image{current_index}.png')
             img.save(img_path)
             f.write(f'image{current_index}.png {lbl}\n')
             current_index += 1


if __name__ == "__main__":
    # 번호판 문자열 생성
    number_strings = generate_number_strings(NUM_IMAGES_TO_SAVE)

    # 생성된 번호판 문자열을 기반으로 합쳐진 문자열 생성
    all_combinations = [f"{number}" for number in tqdm(number_strings, desc="Creating Combinations")]
    print("all_combinations 길이:", len(all_combinations))

    # 이미지 생성 및 저장
    generate_and_save_images(OUTPUT_DIR, NUM_IMAGES_TO_SAVE, all_combinations)
    print("Finished.")