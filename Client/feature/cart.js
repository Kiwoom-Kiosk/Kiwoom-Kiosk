import { SHOPPING_LIST } from "./shoppingList.js";

function addProductToCart(product) {
  const cartItem = JSON.parse(localStorage.getItem("cartItem")) || [];
  cartItem.push(product);
  localStorage.setItem("cartItem", JSON.stringify(cartItem));
}

function addProductAlert(products) {
  const productCards = document.querySelectorAll(".product_card");

  productCards.forEach((productCard, index) => {
    productCard.addEventListener("click", function () {
      const product = products[index];

      const response = confirm("장바구니에 담을까요?");
      if (response) {
        alert("장바구니에 담겼습니다.");
        addProductToCart(product);
      } else {
        alert("취소됐습니다.");
      }
    });
  });
}

const navLists = document.querySelectorAll(".menu_nav_li");

navLists.forEach((navList) => {
  navList.addEventListener("click", function (event) {
    const clickedCategory = event.target.textContent.toLowerCase();
    const filteredProducts = SHOPPING_LIST.filter(
      (product) => product.category.toLowerCase() === clickedCategory
    );
    addProductAlert(filteredProducts);
  });
});

const navCart = document.querySelector(".goCart");
navCart.addEventListener("click", () => {
  window.location.href = "../pages/cart.html";
});

addProductAlert(SHOPPING_LIST);
