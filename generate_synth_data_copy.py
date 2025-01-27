import numpy as np
import random
from trdg.generators import GeneratorFromStrings
from tqdm.auto import tqdm
import os
import pandas as pd

# 사용할 한글 리스트 (일부 글자만 사용)
korean_chars = ["가","거","고","구","나","너","노","누","다","더","도","두","라","러","로","루","마","머","모","무","버","보","부","서","소","수","어","오","우","저","조","주","하","허","호"]
# 사용할 지역 한글 리스트 (예시)
region_korean_chars = ["서울", "경기", "인천", "강원", "충남", "충북", "대전", "경북", "경남", "대구", "울산", "부산", "광주", "전남", "전북", "제주"]

NUM_IMAGES_TO_SAVE = 1000

# #helper funcs and data to generate images
# df = pd.read_csv("trucknumber_database.csv", on_bad_lines='skip', sep='\t', low_memory=True)
# all_words = df[["number"]].to_numpy().flatten()

number_strings = []

# 70%는 지역 번호판 (지역 한글 2자리 + 숫자 2~3자리 + 한글 1자리 + 숫자 4자리)
for i in range(NUM_IMAGES_TO_SAVE* 7 // 10):
    region_char = random.choice(region_korean_chars)  # 지역 한글
    front_digits = np.random.randint(10, 1000)
    if random.random() < 0.5:
        front_digits = np.random.randint(10, 100)  # 숫자 2자리
    korean_char = random.choice(korean_chars)  # 한글 1자리
    back_digits = np.random.randint(1000, 10000)  # 숫자 4자리
    number_string = f"{region_char}{front_digits}{korean_char}{back_digits}"
    number_strings.append(number_string)

# 20%는 신형 번호판 (숫자 3자리 + 한글 1자리 + 숫자 4자리)
for i in range(NUM_IMAGES_TO_SAVE * 2 // 10):
    front_digits = np.random.randint(100, 1000)  # 3자리 숫자 (100~999)
    korean_char = random.choice(korean_chars)  # 한글 1자리
    back_digits = np.random.randint(1000, 10000)  # 4자리 숫자 (1000~9999)
    number_string = f"{front_digits}{korean_char}{back_digits}"
    number_strings.append(number_string)

# 10%는 구형 번호판 (숫자 2자리 + 한글 1자리 + 숫자 4자리)
for i in range(NUM_IMAGES_TO_SAVE * 1 // 10):
    front_digits = np.random.randint(10, 100)  # 2자리 숫자 (10~99)
    korean_char = random.choice(korean_chars)  # 한글 1자리
    back_digits = np.random.randint(1000, 10000)  # 4자리 숫자 (1000~9999)
    number_string = f"{front_digits}{korean_char}{back_digits}"
    number_strings.append(number_string)

#now given word list and number list, get all combinations
all_combinations = []
for number in tqdm(number_strings):
    combined_string = f"{number}"
    all_combinations.append(combined_string)   

print("all_combinations 길이:",len(all_combinations))
NUM_IMAGES_TO_SAVE = min(len(all_combinations), NUM_IMAGES_TO_SAVE) # 두 값중 작은 값으로 설정

# 폰트 파일 경로 설정 (실제 폰트 파일 경로로 변경해야 함)
font_path = "C:/04. AI Project/github/font/NanumGothic.ttf"  # 실제 폰트 파일 경로로 변경

# 이미지 생성기 생성 (이제 문자열 리스트를 올바르게 전달)
generator = GeneratorFromStrings(
    strings=all_combinations,  # 문자열 리스트 전달
    background_type=0,
    count=NUM_IMAGES_TO_SAVE,
    # font_path=font_path,
    language='ko'
)
# save images from generator
# if output folder doesnt exist, create it
if not os.path.exists('output'):
    os.makedirs('output')
#if labels.txt doesnt exist, create it
if not os.path.exists('output/labels.txt'):
    f = open("output/labels.txt", "w", encoding='utf-8')  # UTF-8 인코딩 지정
    f.close()

#open txt file
current_index = 0
f = open("output/labels.txt", "a", encoding='utf-8') # UTF-8 인코딩 지정

for counter, (img, lbl) in tqdm(enumerate(generator), total = NUM_IMAGES_TO_SAVE):
    if (counter >= NUM_IMAGES_TO_SAVE):
        break
    #save pillow image
    img.save(f'output/image{current_index}.png')
    f.write(f'image{current_index}.png {lbl}\n')
    current_index += 1

f.close()
