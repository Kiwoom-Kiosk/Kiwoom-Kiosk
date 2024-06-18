# Kiwoom-Kiosk
🌱 [키움 키오스크] 디지털 약자를 위한 AI 화면 키움 키오스크



## OpenSource Programming Team Project 

| 프로젝트 소개 | 주요 기능 | 제작기간 | 사용 스택 |
|:-------------:|:---------:|:--------:|:--------:|
|"키움 키오스크"<br>디지털 약자를 위한<br>AI 화면 키움 키오스크 |딥러닝_연령층 이미지 인식<br>프론트_키오스크 화면 제작<br>백_키오스크 기능 구현|2024.05.20 ~ 2024.06.19|<img src="https://img.shields.io/badge/Google Colab-F9AB00?style=for-the-badge&logo=Google Colab&logoColor=white"><br><img src="https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=JavaScript&logoColor=white"> <img src="https://img.shields.io/badge/React-61DAFB?style=for-the-badge&logo=React&logoColor=white"><br><img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white"> <img src="https://img.shields.io/badge/django-092E20?style=for-the-badge&logo=django&logoColor=white"><br><img src="https://img.shields.io/badge/github-181717?style=for-the-badge&logo=github&logoColor=white">  |

<br></br>
## Django 코드 구성
⛓️ Django와 CNN 모델을 연결하여 이미지에서 나이를 예측하는 웹 애플리케이션을 구성
<br></br>

## 1️⃣ views.py
### 1. 사용자 정의 손실 함수 정의

```python
def custom_mse(y_true, y_pred):
    return MeanSquaredError()(y_true, y_pred)
```

`custom_mse` 
- 평균 제곱 오차(MSE)를 계산하고, 기본적으로 제공되는 `MeanSquaredError` 클래스를 사용하여 구현
- 추후 모델을 컴파일할 때 사용
<br></br>

### 2. 모델 로드 시 custom_objects 인수 사용

```python
model = load_model(settings.MODEL_PATH, custom_objects={'mse': custom_mse})
```

`load_model` 
- 미리 학습된 모델을 로드하고 `settings.MODEL_PATH`는 모델 파일의 경로를 나타내며, `custom_objects` 인수를 통해 사용자 정의 손실 함수인 `custom_mse`를 모델에 추가
- 모델을 로드하면서 사용자 정의 손실 함수를 정의된 이름('mse')으로 인식하도록 구현
<br></br>

### 3. 모델의 입력 크기 확인 및 이미지 전처리 함수

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

### 4. home 함수 및 upload_image 함수

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

## 2️⃣ settings.py

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
