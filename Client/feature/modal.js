const modal = document.querySelector(".modal");
const openModalButton = document.querySelector("#open_modal");
const closeModalButton = document.querySelector("#close_modal");

openModalButton.addEventListener("click", showModal);
closeModalButton.addEventListener("click", hideModal);

function showModal() {
  modal.classList.remove("hidden");
}

function hideModal() {
  modal.classList.add("hidden");
}
