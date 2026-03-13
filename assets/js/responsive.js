// #######################
// RESPONSIVE PAGE HANDLER
// #######################

import { enableNavbar, disableNavbar } from "./toggle_menu.js";

export function handleMediaQuery() {
    const mainElement = document.querySelector("main");
    const mainContainerElement = document.querySelector("div.main-container");
    const asideElement = document.querySelector("aside");
    const windowWidth = window.innerWidth;
    if (windowWidth < 595) {
        enableNavbar();
        mainElement.style.marginLeft = '';
        mainElement.style.marginRight = '';
        mainElement.style.width = '';
        mainElement.style.paddingRight = '0';
        mainElement.style.paddingLeft = '';
        mainContainerElement.style.borderRight = "none";
        mainContainerElement.style.borderTopRightRadius = '0';
        mainContainerElement.style.borderBottomRightRadius = '0';
        mainContainerElement.style.borderLeft = "var(--cntnr-brdr-thcknss) solid black";
        mainContainerElement.style.borderTopLeftRadius = "10px";
        mainContainerElement.style.borderBottomLeftRadius = "10px";
        if (!asideElement) {
            mainElement.style.marginRight = '0';
            mainElement.style.marginLeft = '0';
            mainElement.style.width = "100%";
            mainElement.style.paddingLeft = '0';
            mainContainerElement.style.borderLeft = "none";
            mainContainerElement.style.borderTopLeftRadius = '0';
            mainContainerElement.style.borderBottomLeftRadius = '0';
        }
    } else if (windowWidth < 1024) {
        disableNavbar();
        mainElement.removeAttribute("style");
        mainContainerElement.removeAttribute("style");
        if (!asideElement) {
            mainElement.style.marginLeft = "5%";
            mainElement.style.width = "90%";
            mainElement.style.paddingRight = "3rem";
            mainElement.style.paddingLeft = "3rem";
            mainContainerElement.style.border = "var(--cntnr-brdr-styl)";
            mainContainerElement.style.borderRadius = "var(--cntnr-brdr-rds)";
        }
    } else if (windowWidth < 2000 || 2000 <= windowWidth) {
        mainElement.removeAttribute("style");
        if (!asideElement) {
            mainElement.style.marginLeft = "20%";
            mainElement.style.width = "60%";
            mainElement.style.paddingLeft = "3rem";
        }
    }
}
