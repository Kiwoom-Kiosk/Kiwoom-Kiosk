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

<br/>

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
<img width="580" src="https://github.com/Kiwoom-Kiosk/Kiwoom-Kiosk/assets/114728629/55c27ff0-e27d-4d89-9466-bc701f450ec7">

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

<br><br/>

## Backend
⛓️ Django와 CNN 모델을 연결하여 이미지에서 나이를 예측하는 웹 애플리케이션을 구성
<br></br>

### 1️⃣ views.py
#### 1. 사용자 정의 손실 함수 정의

```python
def custom_mse(y_true, y_pred):
    return MeanSquaredError()(y_true, y_pred)
```

`custom_mse` 
- 평균 제곱 오차(MSE)를 계산하고, 기본적으로 제공되는 `MeanSquaredError` 클래스를 사용하여 구현
- 추후 모델을 컴파일할 때 사용
<br></br>

#### 2. 모델 로드 시 custom_objects 인수 사용

```python
model = load_model(settings.MODEL_PATH, custom_objects={'mse': custom_mse})
```

`load_model` 
- 미리 학습된 모델을 로드하고 `settings.MODEL_PATH`는 모델 파일의 경로를 나타내며, `custom_objects` 인수를 통해 사용자 정의 손실 함수인 `custom_mse`를 모델에 추가
- 모델을 로드하면서 사용자 정의 손실 함수를 정의된 이름('mse')으로 인식하도록 구현
<br></br>

#### 3. 모델의 입력 크기 확인 및 이미지 전처리 함수

```python
input_shape = model.layers[0].input_shape
target_size = input_shape[1:3]

def load_and_preprocess_image(filepath, target_size=(200, 200)):
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img.astype('float32') / 255.0  # 스케일링
    return img
```

- `input_shape`: 모델의 첫 번째 레이어의 입력 형태를 확인하는 변수로, 모델에 입력할 이미지의 크기를 결정하는 데 사용
- `target_size`: 이미지 전처리 시 리사이징할 크기를 지정하고, 모델의 입력 형태와 일치시키기 위해 사용
- `load_and_preprocess_image`: 이미지를 전처리하는 함수이고, OpenCV를 사용하여 이미지를 읽고 RGB로 변환한 후, 지정된 크기로 리사이즈하며, 0에서 1 사이의 값으로 정규화하도록 구현
<br></br>

#### 4. home 함수 및 upload_image 함수

```python
def home(request):
    return render(request, 'home.html')

def upload_image(request):
    if request.method == 'POST' and request.FILES['image']:
        img = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save(img.name, img)
        uploaded_file_url = fs.url(filename)

        # 이미지 전처리 및 예측
        img_path = os.path.join(settings.MEDIA_ROOT, filename)
        img = load_and_preprocess_image(img_path, target_size=target_size)
        img_array = np.expand_dims(img, axis=0)  # 배치 차원 추가

        prediction = model.predict(img_array)
        age = int(prediction[0][0])

        return render(request, 'result.html', {
            'uploaded_file_url': uploaded_file_url,
            'age': age
        })
    return redirect('home')
```

- `home 함수`: 홈 페이지를 렌더링
- `upload_image 함수`: POST 요청이 오면 이미지를 업로드하고 `load_and_preprocess_image` 함수를 통해 이미지를 전처리한 후, 전처리된 이미지를 모델에 입력하여 나이를 예측하고, 예측 결과를 `result.html` 템플릿에 전달하여 사용자에게 제공
<br></br>

### 2️⃣ settings.py

```python
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent

# 모델 파일 경로 설정
MODEL_PATH = os.path.join(BASE_DIR, 'age_model_vgg16.h5')

# Django 기본 설정은 생략, 추가된 부분 설명
```

- **MODEL_PATH**: CNN 모델 파일(`age_model_vgg16.h5`)의 절대 경로를 설정하고, 해당 경로로 모델을 `load_model` 함수로 로드
- **MEDIA_ROOT 및 MEDIA_URL**: `settings.py`에서 `MEDIA_ROOT`는 업로드된 파일의 저장 경로를 설정하고, `MEDIA_URL`은 업로드된 파일에 접근할 수 있는 URL을 설정하도록 구현
- **전체적인 동작 흐름**
  - 사용자가 이미지를 업로드하면, 이는 `upload_image` 함수에서 처리하도록 구현 이미지가 업로드되면 해당 이미지를 전처리하고, CNN 모델에 입력하여 나이를 예측한 후, 결과를 사용자에게 보여주는 방식으로 동작
 
<br></br>

## Frontend
### 📌 폴더 구조

```
|-- 📁 CLIENT
|    |
|        |-- 📁 asset 
|    |      |
|    |      |-- 📁 icon
|    |      |-- 📁 img
|    |
|        |-- 📁 feature 
|    |      |-- cart.js // 장바구니 추가 기능 
|    |      |-- filter.js // home.html -> 메뉴 카테고리 선택에 따른 필터 기능 
|    |      |-- shoppingList.js //  상수 데이터 리스트 
|    |      
|    | 📁 pages
|    |      | - predictor.html // 연령 예측 page
|    |      | - home.html // home page
|    |      
|    | 📁 Styles
|    |      | - predictor.css// 연령 예측기 page css
|    |      | - home_Eldery.css// 키오스크 디지털 약자  뷰 css
|    |      | - home._Youth.css // 키오스크 일반 뷰 css


```

<br/>

### 📌 화면 구현 기능

#### 1️⃣ 파일 업로드 및 api 통신

```html
      document.getElementById("uploadForm").onsubmit = async function (event) {
        event.preventDefault();
        const formData = new FormData();
        const fileInput = document.getElementById("fileInput");
        formData.append("file", fileInput.files[0]);

        const uploadedImage = document.getElementById("uploadedImage");
        uploadedImage.src = URL.createObjectURL(fileInput.files[0]);
        uploadedImage.style.display = "block";

        const result = await uploadFile(formData);
        if (result.age !== undefined) {
          const age = result.age;
        } 
      };
```
- **동작**: 파일을 업로드하고, `FormData` 객체를 생성하여 파일을 추가합니다. 이후 `uploadFile` 함수를 사용하여 서버에 파일을 전송하고, 연령 예측 결과를 받아옴
- **기능**: 파일을 업로드한 후에는 업로드된 이미지를 화면에 표시하고, 예측된 연령을 받아서 추가적인 처리 진행

<br/>

#### 2️⃣ 연령 예측에 따른 청년/고령 분류 결과를 setElderyStatus로 전달

```html
      function goToNextPage() {
        const fileInput = document.getElementById("fileInput");
        const result = document.getElementById("result");
        const age = result.age;
        if (age >= 50) {
         setElderyStatus(true);
        } else {
          setElderyStatus(false);
        }
      }
```
- **동작**: `isEldery`라는 boolean 변수를 받아서, 이를 기반으로 다음 페이지 URL을 생성하고, 해당 URL로 페이지를 이동
- **기능**: 연령 예측 결과에 따라 청년과 고령을 분류하고, 그에 따른 다음 페이지로의 자동 이동을 담당

<br/>

#### 3️⃣ 페이지 내 예측된 연령 출력

```html
      function displayAge() {
        const fileInput = document.getElementById("fileInput");
        const result = document.getElementById("result");
        const age = result.age;
        if (age >= 50) {
          result.innerHTML = '<div class="large-ui">예측 연령: ' + age + "</div>";
        } else {
          result.innerHTML = '<div class="regular-ui">예측 연령: ' + age + "</div>";
        }
      }
```

- **동작**: 예측된 연령을 표시하는 함수로, `result` 엘리먼트에 연령에 따라 다른 스타일의 텍스트를 삽입
- **기능**: 예측된 연령을 사용자에게 시각적으로 제공. 예측된 연령이 50세 이상이면 큰 텍스트 스타일을, 그렇지 않으면 일반적인 텍스트 스타일을 적용

<br/>

| <img width="310" src="https://github.com/Kiwoom-Kiosk/Kiwoom-Kiosk/assets/114728629/818c704d-d13f-4db6-9ee3-1d485cbb616f"> | <img width="310" src="https://github.com/Kiwoom-Kiosk/Kiwoom-Kiosk/assets/114728629/7deee988-469f-4914-9e99-12a775414c90"> |
|:------:|:------:|
| **50세 미만 사용자 연령 예측** | **50세 이상 사용자 연령 예측** |

<br/>

#### 4️⃣ 연령 예측에 따른 청년/고령 분류 결과를 isEldery변수의 boolean 값으로 입력 받아 분류에 따른 키오스크 페이지 이동

```html
      function setElderyStatus(isEldery) {
        var nextPageUrl = "/pages/home.html";
        nextPageUrl += "?isEldery=" + encodeURIComponent(isEldery);

        window.location.href = nextPageUrl;
      }
```

- **동작**: 예측된 연령을 기반으로 `setElderyStatus` 함수를 호출하여 청년과 고령을 분류하고, 그 결과에 따라 다음 페이지로 이동
- **기능**: 연령 예측 결과를 받아서 청년과 고령을 분류하고, 이를 기반으로 다음 단계의 키오스크 페이지로 사용자를 이동

<br/>

| <img width="310" src="https://github.com/Kiwoom-Kiosk/Kiwoom-Kiosk/assets/114728629/52da1c18-e728-4b24-8453-d263f194e8b4"> | <img width="320" src="https://github.com/Kiwoom-Kiosk/Kiwoom-Kiosk/assets/114728629/b76b2924-b195-47dd-bbfb-87cea075c252"> |
|:------:|:------:|
| **50세 미만 사용자 키오스크 메뉴** | **50세 이상 사용자 키오스크 메뉴** |


