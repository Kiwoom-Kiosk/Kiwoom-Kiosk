# Kiwoom-Kiosk
🌱 [키움 키오스크] 디지털 약자를 위한 AI 화면 키움 키오스크



## OpenSource Programming Team Project 

| 프로젝트 소개 | 주요 기능 | 제작기간 | 사용 스택 |
|:-------------:|:---------:|:--------:|:--------:|
|"키움 키오스크"<br>디지털 약자를 위한<br>AI 화면 키움 키오스크 |딥러닝_연령층 이미지 인식<br>프론트_키오스크 화면 제작<br>백_키오스크 기능 구현|2024.05.20 ~ 2024.06.19|<img src="https://img.shields.io/badge/Google Colab-F9AB00?style=for-the-badge&logo=Google Colab&logoColor=white"><br><img src="https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=JavaScript&logoColor=white"> <img src="https://img.shields.io/badge/React-61DAFB?style=for-the-badge&logo=React&logoColor=white"><br><img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white"> <img src="https://img.shields.io/badge/django-092E20?style=for-the-badge&logo=django&logoColor=white"><br><img src="https://img.shields.io/badge/github-181717?style=for-the-badge&logo=github&logoColor=white">  |


## 📌 폴더 구조

```
|-- 📁 CLIENT
|	 |
|        |-- 📁 asset 
|	 |		|
|	 |		|-- 📁 icon
|	 |		|-- 📁 img
|	 |
|        |-- 📁 feature 
|	 |		|-- cart.js // 장바구니 추가 기능 
|	 |		|-- filter.js // home.html -> 메뉴 카테고리 선택에 따른 필터 기능 
|	 |		|-- shoppingList.js //  상수 데이터 리스트 
|	 |		
|	 | 📁 pages
|	 |		| - predictor.html // 연령 예측 page
|	 |		| - home.html // home page
|	 |		
|	 | 📁 Styles
|	 |		| - predictor.css// 연령 예측기 page css
|	 |		| - home_Eldery.css// 키오스크 디지털 약자  뷰 css
|	 |		| - home._Youth.css // 키오스크 일반 뷰 css


```

---

## 📌 주요 기능 

- 파일 업로드 및 api 통신

```
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

- 연령 예측에 따른 청년/고령 분류 결과를 isEldery변수의 boolean 값으로 입력 받아 분류에 따른 키오스크 페이지 이동

```
      function setElderyStatus(isEldery) {
        var nextPageUrl = "/pages/home.html";
        nextPageUrl += "?isEldery=" + encodeURIComponent(isEldery);

        window.location.href = nextPageUrl;
      }
```

- 페이지 내 예측된 연령 출력

```
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

- 연령 예측에 따른 청년/고령 분류 결과를 setElderyStatus로 전달

```
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

## 📌스크린샷 
![image](https://github.com/Kiwoom-Kiosk/Kiwoom-Kiosk/assets/105477246/b7af6d84-b8b9-4e23-95e8-9c63997bfb87)
![image](https://github.com/Kiwoom-Kiosk/Kiwoom-Kiosk/assets/105477246/55fa6030-77c0-4fce-b73a-1da70ece49a5)
![image](https://github.com/Kiwoom-Kiosk/Kiwoom-Kiosk/assets/105477246/b5ac7576-5eac-4555-9360-109427ce5eb7)
![image](https://github.com/Kiwoom-Kiosk/Kiwoom-Kiosk/assets/105477246/51bf075f-f29b-4c85-9c23-a09b3c29905e)
