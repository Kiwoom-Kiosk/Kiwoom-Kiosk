# from django.shortcuts import render
# from django.http import JsonResponse
# from django.core.files.storage import default_storage
# from django.conf import settings
# import os
# import cv2
# import numpy as np
# from tensorflow.keras.models import load_model

# # 모델 로드
# model_path = os.path.join(settings.BASE_DIR, 'age_model_3.h5')
# age_model = load_model(model_path)

# # 얼굴 검출기 로드
# face_cascade_path = os.path.join(settings.BASE_DIR, 'haarcascade_frontalface_default.xml')
# face_cascade = cv2.CascadeClassifier(face_cascade_path)

# def index(request):
#     return render(request, 'predictor/index.html')

# def predict_age(image_path):
#     pic = cv2.imread(image_path)
#     faces = face_cascade.detectMultiScale(pic, scaleFactor=1.11, minNeighbors=8)
#     if len(faces) == 0:
#         return None
#     (x, y, w, h) = faces[0]
#     img = pic[y:y + h, x:x + w]
#     img = cv2.resize(img, (200, 200))
#     age_predict = age_model.predict(np.array(img).reshape(-1, 200, 200, 3))
#     return int(age_predict[0][0])

# def upload_image(request):
#     if request.method == 'POST' and request.FILES['file']:
#         file = request.FILES['file']
#         file_path = default_storage.save('uploads/' + file.name, file)
#         full_file_path = os.path.join(settings.MEDIA_ROOT, file_path)
#         age = predict_age(full_file_path)
#         if age is not None:
#             return JsonResponse({'age': age})
#         else:
#             return JsonResponse({'error': 'No face detected'}, status=400)
#     return JsonResponse({'error': 'Invalid request'}, status=400)

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the model and the cascade classifier
age_model = load_model('age_model_3.h5')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def index(request):
    return render(request, 'predictor/index.html')

@csrf_exempt
def upload(request):
    if request.method == 'POST' and request.FILES['file']:
        file = request.FILES['file']
        img = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        
        faces = face_cascade.detectMultiScale(img, scaleFactor=1.11, minNeighbors=8)
        image_size = 200
        age_ = []
        for (x, y, w, h) in faces:
            face_img = img[y:y + h, x:x + w]
            face_img = cv2.resize(face_img, (image_size, image_size))
            age_predict = age_model.predict(np.array(face_img).reshape(-1, image_size, image_size, 3))
            age_.append(int(age_predict[0][0]))

        if age_:
            return JsonResponse({'age': age_[0]})
        else:
            return JsonResponse({'error': 'No face detected'}, status=400)
    
    return JsonResponse({'error': 'Invalid request'}, status=400)
