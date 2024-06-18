import { SHOPPING_LIST } from "./shoppingList.js";

const shoppingListData = SHOPPING_LIST;

// 모든 상품 데이터
const allProducts = shoppingListData.map(({ name, imageUrl, value }) => ({
  name,
  imageUrl,
  value,
}));

// 모든 상품을 먼저 보여줌
showProducts(allProducts);

// 상품 카드 생성
function createProductCard(name, imageUrl, value) {
  const productMenu = document.createElement("div");
  productMenu.classList.add("product_card");

  const productInfo = document.createElement("ul");
  productInfo.classList.add("product");

  const image = document.createElement("img");
  image.classList.add("product_img");
  image.src = imageUrl;
  image.alt = name;

  const heartIcon = document.createElement("i");
  heartIcon.classList.add("fa-solid", "fa-heart");
  heartIcon.alt = "좋아요 버튼";

  const productName = document.createElement("li");
  productName.textContent = name;

  const productvalue = document.createElement("li");
  productvalue.textContent = value ? value.toLocaleString() : "가격 정보 없음";

  productInfo.appendChild(image);
  productInfo.appendChild(heartIcon);
  productInfo.appendChild(productName);
  productInfo.appendChild(productvalue);

  productMenu.appendChild(productInfo);
  return productMenu;
}

// 상품 목록을 보여주는 함수
function showProducts(products) {
  const resultContainer = document.querySelector(".product_section");
  resultContainer.innerHTML = "";

  products.forEach((product) => {
    const { name, imageUrl, value } = product;
    const productCard = createProductCard(name, imageUrl, value);
    resultContainer.appendChild(productCard);
  });
}

// 카테고리 필터링
const navLists = document.querySelectorAll(".menu_nav_li");

navLists.forEach((navList) => {
  navList.addEventListener("click", function (event) {
    const clickedCategory = event.target.textContent.toLowerCase();
    if (clickedCategory == "all") {
      const filteredProducts = allProducts;
      showProducts(filteredProducts);
    } else {
      const filteredProducts = shoppingListData.filter(
        (product) => product.category.toLowerCase() === clickedCategory
      );
      showProducts(filteredProducts);
    }
  });
});
