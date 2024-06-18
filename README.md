# Kiwoom-Kiosk
🌱 [키움 키오스크] 디지털 약자를 위한 AI 화면 키움 키오스크

<br/>

## OpenSource Programming Team Project 

| 프로젝트 소개 | 주요 기능 | 제작기간 | 사용 스택 |
|:-------------:|:---------:|:--------:|:--------:|
|"키움 키오스크"<br>디지털 약자를 위한<br>AI 화면 키움 키오스크 |딥러닝_연령층 이미지 인식<br>프론트_키오스크 화면 제작<br>백_키오스크 기능 구현|2024.05.20 ~ 2024.06.19|<img src="https://img.shields.io/badge/Google Colab-F9AB00?style=for-the-badge&logo=Google Colab&logoColor=white"><br><img src="https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=JavaScript&logoColor=white"> <img src="https://img.shields.io/badge/React-61DAFB?style=for-the-badge&logo=React&logoColor=white"><br><img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white"> <img src="https://img.shields.io/badge/django-092E20?style=for-the-badge&logo=django&logoColor=white"><br><img src="https://img.shields.io/badge/github-181717?style=for-the-badge&logo=github&logoColor=white">  |

<br/>

## 📌 폴더 구조

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

## 📌 화면 구현 기능

### 1️⃣ 파일 업로드 및 api 통신

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

### 2️⃣ 연령 예측에 따른 청년/고령 분류 결과를 setElderyStatus로 전달

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

### 3️⃣ 페이지 내 예측된 연령 출력

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

### 4️⃣ 연령 예측에 따른 청년/고령 분류 결과를 isEldery변수의 boolean 값으로 입력 받아 분류에 따른 키오스크 페이지 이동

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

