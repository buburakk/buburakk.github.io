// ###############################
// TOGGLE NAVIGATION BAR ANIMATION
// ###############################

const burgerMenuElement = document.getElementById("toggleMenuIcon");
const toggleMenuElements = document.querySelector("ul.toggle-menu-elements");
let navbarVisible = false;

function showNavbar() {
    toggleMenuElements.style.display = "block";
}

function hideNavbar() {
    toggleMenuElements.style.display = "none";
    navbarVisible = false;
}

function onBurgerClick(event) {
    event.stopPropagation();
    navbarVisible = !navbarVisible;
    // Alternative use of the condition below
    // navbarVisible ? showNavbar() : hideNavbar();
    if (navbarVisible) {
        showNavbar();
    } else {
        hideNavbar();
    }
}

function onDocumentClick(event) {
    if (navbarVisible &&
        !burgerMenuElement.contains(event.target) &&
        !toggleMenuElements.contains(event.target)
    ) {
        hideNavbar();
    }
}

export function enableNavbar() {
    disableNavbar();
    burgerMenuElement.addEventListener("click", onBurgerClick);
    document.addEventListener("click", onDocumentClick);
    hideNavbar();
}

export function disableNavbar() {
    burgerMenuElement.removeEventListener("click", onBurgerClick);
    document.removeEventListener("click", onDocumentClick);
    // Can be removed safely, but still wait
    // toggleMenuElements.style.display = '';
    toggleMenuElements.removeAttribute("style");
    navbarVisible = false;
}
