# Kiwoom-Kiosk
🌱 [키움 키오스크] 디지털 약자를 위한 AI 화면 키움 키오스크

</br>

## OpenSource Programming Team Project 

| 프로젝트 소개 | 주요 기능 | 제작기간 | 사용 스택 |
|:-------------:|:---------:|:--------:|:--------:|
|"키움 키오스크"<br>디지털 약자를 위한<br>AI 화면 키움 키오스크 |딥러닝_연령층 이미지 인식<br>프론트_키오스크 화면 제작<br>백_키오스크 기능 구현|2024.05.20 ~ 2024.06.19|<img src="https://img.shields.io/badge/Google Colab-F9AB00?style=for-the-badge&logo=Google Colab&logoColor=white"><br><img src="https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=JavaScript&logoColor=white"> <img src="https://img.shields.io/badge/React-61DAFB?style=for-the-badge&logo=React&logoColor=white"><br><img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white"> <img src="https://img.shields.io/badge/django-092E20?style=for-the-badge&logo=django&logoColor=white"><br><img src="https://img.shields.io/badge/github-181717?style=for-the-badge&logo=github&logoColor=white">  |

</br>

## 👥 팀원 소개

| 🦅[김지수](https://github.com/KimJisu-IT)🦅 | 🐻[박상아](https://github.com/Ivoryeee)🐻 | 🐳[양다연](https://github.com/dayeon1201)🐳 |
|------|---|---|
| Deeplearning / BE | Deeplearning / FE | Deeplearning / BE |


<br/>

## 시연 영상
| ![ezgif-2-e7f3c45a0f](https://github.com/Kiwoom-Kiosk/Kiwoom-Kiosk/assets/114728629/2406fa6b-95b9-4c88-98ef-f0c16b0eb1ef) | ![ezgif-2-cf3e18e03f](https://github.com/Kiwoom-Kiosk/Kiwoom-Kiosk/assets/114728629/cd2469ec-268f-4754-a998-82fa2da053f1) |
|:------:|:------:|
| **50세 미만 사용자 UI** | **50세 이상 사용자 UI** |


## 프로젝트 개요

### 1️⃣ 기술개발 배경
- 디지털 전환과 온라인 서비스 확산으로 인해 다양한 연령대의 사람들이 키오스크와 같은 디지털 기기를 사용하는 빈도가 증가하고 있음.
- 고령자나 디지털 약자들은 작은 텍스트와 복잡한 UI 때문에 디지털 기기 사용에 어려움을 겪고 있음.
    - 이러한 문제를 해결하기 위해 연령대에 맞춘 사용자 인터페이스(UI)를 제공하는 기술이 필요함.
    - 인공지능과 안면 인식 기술을 활용하여 연령대를 예측하고, 이에 맞춰 UI의 텍스트 볼륨 크기를 자동으로 조절하는 시스템이 효과적인 대안이 될 수 있음.

<br/>

### 2️⃣ 기술개발 목표
- 본 연구개발에서는 안면 인식 기반의 연령대 예측 기술을 개발하고, 이를 바탕으로 키오스크 UI의 텍스트 크기를 자동 조절하는 시스템을 구축하는 것을 목표로 함.
- 추가적으로 사용자의 연령대에 따라 최적화된 UI 요소를 정의하고, 사용자 경험을 개선하고자 함.

<br/>

### 3️⃣ 기대효과
#### 📱 [기술적 기대효과]
- 인공지능 기반 안면 인식 및 연령대 예측 기술 확보.
- 연령대 맞춤형 UI 자동 조절 기술을 통해 다양한 사용자에게 적합한 키오스크 환경 제공.
- 다양한 디지털 기기 및 서비스에 적용 가능한 맞춤형 인터페이스 기술 확보.

#### 🧩 [사회적 기대효과]
- 연령대에 맞춘 사용자 경험 제공을 통해 디지털 약자의 디지털 기기 접근성 향상.
- 고령자 및 디지털 약자의 사회적·관계적 고립감 완화 및 디지털 소외 방지.
- 사용자의 만족도 및 편의성 증대를 통한 디지털 서비스의 이용률 향상.

<br/>

### 4️⃣ 서비스 아키텍처 계획
![KakaoTalk_Photo_2024-06-19-00-51-30](https://github.com/Kiwoom-Kiosk/Kiwoom-Kiosk/assets/114728629/d9f2d5c2-0f18-4be0-a267-d2acfa578b5c)

<br/>

### 5️⃣ 일정계획 간트차트
<img width="580" alt="스크린샷 2024-06-19 오전 12 54 37" src="https://github.com/Kiwoom-Kiosk/Kiwoom-Kiosk/assets/114728629/55c27ff0-e27d-4d89-9466-bc701f450ec7">

<br><br/>

## CNN 모델 구현
🔗[Kiwoom-Kiosk 나이 예측 모델 상세보기](https://github.com/Kiwoom-Kiosk/Kiwoom-Kiosk/blob/feature/2-Age-Prediction/Kiwoon-Kiosk.ipynb)
<br><br/>

### 📌 숫자로 표현된 인종, 성별 데이터를 문자열로 변환 / 데이터셋 폴더에서 파일명을 분석하여 나이, 성별, 인종 정보 추출
````python
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
````
<img width="600" alt="스크린샷 2024-06-19 오전 1 11 34" src="https://github.com/Kiwoom-Kiosk/Kiwoom-Kiosk/assets/114728629/c1807641-4dd1-496f-929c-fa4682e084d3">

<br><br/>


### 📌 이미지 전처리, 모델 구성 및 학습
````python
def load_and_preprocess_image(filepath, target_size=(200, 200)):
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    return img

files = df['file'].tolist()
ages = df['age'].tolist()

images = [load_and_preprocess_image(file) for file in files]
age = np.array(ages, dtype=np.int64)
images = np.array(images)

x_train_age, x_test_age, y_train_age, y_test_age = train_test_split(images, age, random_state=42, test_size=0.2)

model = Sequential([
    Flatten(input_shape=(200, 200, 3)),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='linear')
])

model.compile(loss='mse', optimizer=Adam(), metrics=['mae'])

history = model.fit(
    datagen.flow(x_train_age, y_train_age, batch_size=batch_size),
    validation_data=(x_valid_age, y_valid_age),
    epochs=epochs,
    callbacks=callbacks
)

# 학습 결과 시각화
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss')

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.legend()
plt.title('Mean Absolute Error')

plt.show()
````
<img width="624" alt="스크린샷 2024-06-19 오전 1 08 51" src="https://github.com/Kiwoom-Kiosk/Kiwoom-Kiosk/assets/114728629/dad7d270-2764-4058-81bc-089cc5f54464">


<br><br/>

### 📌 테스트 데이터로 모델 평가
````python
loss, mae = model.evaluate(x_test_age, y_test_age, verbose=0)
print(f'Test MAE: {mae}')

new_image = load_and_preprocess_image('test_image.jpg')
predicted_age = model.predict(np.expand_dims(new_image, axis=0))
print(f'Predicted Age: {predicted_age[0][0]}')
````
<img width="600" alt="스크린샷 2024-06-19 오전 1 09 14" src="https://github.com/Kiwoom-Kiosk/Kiwoom-Kiosk/assets/114728629/dee5fbd3-f309-403c-a912-57fad8b3dea7">


