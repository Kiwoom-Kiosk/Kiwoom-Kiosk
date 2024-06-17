from django.shortcuts import render

# Create your views here.
import numpy as np
import os
import cv2
import keras
import tensorflow as tk
from django.conf import settings
from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.losses import MeanSquaredError

# 사용자 정의 손실 함수 정의
def custom_mse(y_true, y_pred):
    return MeanSquaredError()(y_true, y_pred)

# 모델 로드 시 custom_objects 인수 사용
model = load_model(settings.MODEL_PATH, custom_objects={'mse': custom_mse})

# 모델의 입력 크기 확인
input_shape = model.layers[0].input_shape
target_size = input_shape[1:3]

# 이미지 전처리 함수
def load_and_preprocess_image(filepath, target_size=(200, 200)):
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img.astype('float32') / 255.0  # 스케일링
    return img

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