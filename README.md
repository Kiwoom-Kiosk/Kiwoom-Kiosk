# Kiwoom-Kiosk
🌱 [키움 키오스크] 디지털 약자를 위한 AI 화면 키움 키오스크



## OpenSource Programming Team Project 

| 프로젝트 소개 | 주요 기능 | 제작기간 | 사용 스택 |
|:-------------:|:---------:|:--------:|:--------:|
|"키움 키오스크"<br>디지털 약자를 위한<br>AI 화면 키움 키오스크 |딥러닝_연령층 이미지 인식<br>프론트_키오스크 화면 제작<br>백_키오스크 기능 구현|2024.05.20 ~ 2024.06.19|<img src="https://img.shields.io/badge/Google Colab-F9AB00?style=for-the-badge&logo=Google Colab&logoColor=white"><br><img src="https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=JavaScript&logoColor=white"> <img src="https://img.shields.io/badge/React-61DAFB?style=for-the-badge&logo=React&logoColor=white"><br><img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white"> <img src="https://img.shields.io/badge/django-092E20?style=for-the-badge&logo=django&logoColor=white"><br><img src="https://img.shields.io/badge/github-181717?style=for-the-badge&logo=github&logoColor=white">  |

<br></br>

## CNN 모델 구성
🪬 UTKFace 데이터셋을 활용하여 CNN을 구성하고, 나이를 예측하는 모델 학습
<br></br>

#### 💬What: 필요한 라이브러리 설치 및 데이터셋 다운로드

```python
!pip install kaggle
from google.colab import files
files.upload()  # kaggle.json 파일 업로드

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d jangedoo/utkface-new
!unzip utkface-new.zip
```

#### ❗️How:
- **kaggle**: Kaggle API를 사용하고, Kaggle API 토큰인 `kaggle.json` 파일을 업로드
- **kaggle datasets download**: Kaggle에서 UTKFace 데이터셋을 다운로드

<br></br>

#### 💬What: 데이터 준비 및 전처리

```python
import os
import pandas as pd
from glob import glob

# 숫자로 표현된 인종, 성별 데이터를 문자열로 변환하기 위한 딕셔너리 생성
dataset_dict = {
    'race_id': {
        0: 'white',
        1: 'black',
        2: 'asian',
        3: 'indian',
        4: 'others'
    },
    'gender_id': {
        0: 'male',
        1: 'female'
    }
}

# 데이터셋 폴더에서 파일명을 분석하여 나이, 성별, 인종 정보 추출하는 함수
def parse_dataset(dataset_path, ext='jpg'):
    def parse_info_from_file(path):
        try:
            filename = os.path.split(path)[1]
            filename = os.path.splitext(filename)[0]
            age, gender, race, _ = filename.split('_')
            return int(age), dataset_dict['gender_id'][int(gender)], dataset_dict['race_id'][int(race)]
        except Exception as ex:
            return None, None, None

    files = glob(os.path.join(dataset_path, "*.%s" % ext))

    records = []
    for file in files:
        info = parse_info_from_file(file)
        records.append(info)

    df = pd.DataFrame(records)
    df['file'] = files
    df.columns = ['age', 'gender', 'race', 'file']
    df = df.dropna()

    return df

df = parse_dataset('UTKFace')

df.head()
```

#### ❗️How:
- **dataset_dict**: 숫자로 표현된 인종과 성별을 문자열로 변환
- **parse_dataset 함수**: UTKFace 데이터셋 폴더에서 파일을 읽어들여 파일명을 분석하여 나이, 성별, 인종 정보를 추출하고 데이터프레임으로 반환
- **glob**: 파일 시스템 경로명 패턴을 사용하여 파일들을 읽어오기

<br></br>

#### 💬What: 모델 구축 및 학습

```python
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# 이미지 전처리 함수
def load_and_preprocess_image(filepath, target_size=(200, 200)):
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    return img

# 데이터 준비
files = df['file'].tolist()
ages = df['age'].tolist()

images = [load_and_preprocess_image(file) for file in files]
age = np.array(ages, dtype=np.int64)
images = np.array(images)

# 데이터 분할
x_train_age, x_test_age, y_train_age, y_test_age = train_test_split(images, age, random_state=42, test_size=0.2)

# 모델 구성
model = Sequential([
    Flatten(input_shape=(200, 200, 3)),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='linear')
])

# 모델 컴파일
model.compile(loss='mse', optimizer=Adam(), metrics=['mae'])

# 모델 학습
history = model.fit(
    datagen.flow(x_train_age, y_train_age, batch_size=batch_size),
    validation_data=(x_valid_age, y_valid_age),
    epochs=epochs,
    callbacks=callbacks
)
   
```

#### ❗️How:
- **load_and_preprocess_image 함수**: 이미지 파일을 로드하고 RGB 형식으로 변환하며, 지정된 크기로 리사이즈
- **데이터 준비**: UTKFace 데이터셋에서 이미지와 나이 정보를 준비
- **데이터 분할**: `train_test_split` 함수를 사용하여 학습 데이터와 테스트 데이터를 분리
- **모델 구성**: `Sequential` 모델을 사용하여 Flatten, Dense, Dropout 레이어를 쌓고, 최종 출력 레이어는 나이를 예측하기 위한 선형 활성화 함수를 사용
- **모델 컴파일 및 학습**: 손실 함수는 평균 제곱 오차(Mean Squared Error, MSE)를 사용하고, 옵티마이저는 Adam을 사용하여 모델을 컴파일하고 학습 진행

<br></br>

#### 💬What: 모델 평가 및 예측

```python
# 테스트 데이터로 모델 평가
loss, mae = model.evaluate(x_test_age, y_test_age, verbose=0)
print(f'Test MAE: {mae}')

# 새로운 이미지에서 나이 예측
new_image = load_and_preprocess_image('test_image.jpg')
predicted_age = model.predict(np.expand_dims(new_image, axis=0))
print(f'Predicted Age: {predicted_age[0][0]}')
```

#### ❗️How:
- **모델 평가**: `evaluate` 메서드를 사용하여 테스트 데이터에서 모델을 평가하고 평균 절대 오차(Mean Absolute Error, MAE)를 출력
- **새로운 이미지 예측**: 새로운 이미지를 로드하고 전처리한 후, `predict` 메서드를 사용하여 해당 이미지에서의 나이를 예측

<br></br>

### ⚙️ 학습 결과
--------
![KakaoTalk_Photo_2024-06-19-00-08-29](https://github.com/Kiwoom-Kiosk/Kiwoom-Kiosk/assets/114728629/20071170-8654-4c0b-adf9-4c2762290537)

