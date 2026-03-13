// ##############
// LOADER MODULES
// ##############

// import { handleMediaQuery } from "./responsive.js";
import { quoteMap, createLogoContent, 
            createImageContent, createCardContent, 
                createDropdownMenu, createMarkdownContent, 
                    createQuoteContent, createFooterContent } from "./contents.js";
import { renderMathJax } from "./utils.js";

export async function loadLogoContent() {
    const element = document.querySelector("div.header-logo-container");
    element.innerHTML = await createLogoContent();
}

export async function loadHomeImage() {
    const element = document.querySelector("#indexSection h1");
    element.insertAdjacentHTML("beforebegin", await createImageContent());
}

export async function loadCardContent() {
    const element = document.querySelector("#aboutSection h1");
    element.insertAdjacentHTML("beforebegin", await createCardContent());
}

export async function loadDropdownMenu(titles, markdowns) {
    const element = document.querySelector("div.dropdown-menu-container");
    let htmlContent = `
    <nav id="dropdownMenu">
        <ul class="dropdown-menu-elements">
    `;
    for (let i = 0; i < Math.min(titles.length, markdowns.length); i++) {
        htmlContent += await createDropdownMenu(titles[i], markdowns[i]);
    }
    htmlContent += "</ul></nav>";
    element.innerHTML = htmlContent;
}

export async function loadMarkdownContent(titles, markdowns) {
    const element = document.querySelector("div.article-container");
    let htmlContent = '';
    for (let i = 0; i < Math.min(titles.length, markdowns.length); i++) {
        htmlContent += await createMarkdownContent(titles[i], [markdowns[i]]);
    }
    element.innerHTML = htmlContent;
    await renderMathJax(element);
}

export async function loadQuoteContent() {
    const element = document.querySelector("div.quote-container");
    element.innerHTML = await createQuoteContent(quoteMap);
}

export async function loadFooterContent() {
    const element = document.querySelector("div.footer-container");
    element.innerHTML = await createFooterContent();
}

export function loadAboutLayout() {
    window.scrollTo(0, 0);
    const asideElement = document.querySelector("aside");
    if (asideElement) asideElement.remove();
    const mainElement = document.querySelector("main");
    // mainElement.style.marginRight = "12.5%";
    mainElement.style.marginLeft = "12.5%";
    mainElement.style.width = "75%";
    // handleMediaQuery();
}

const defaultAsideElement = document.querySelector("aside").cloneNode(true);

export function loadDefaultLayout() {
    window.scrollTo(0, 0);
    // const mainElement = document.querySelector("main");
    let asideElement = document.querySelector("aside");
    // mainElement.style.margin = '';
    if (!asideElement) {
        const headerElement = document.querySelector("header");
        headerElement.insertAdjacentElement("afterend", defaultAsideElement);
        asideElement = document.querySelector("aside");
    }
    // mainElement.style.width = `
    // calc(100% - ${window.getComputedStyle(asideElement).width} 
    // - ${window.getComputedStyle(asideElement).marginRight} 
    // - ${window.getComputedStyle(mainElement).marginLeft})
    // `;
    // mainElement.style.width = '';
    // handleMediaQuery();
}
