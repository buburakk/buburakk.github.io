//  ################
//  GLOBAL VARIABLES
//  ################

const navigationMap = {
    "index/": ["index.md"],
    "notes/": ["update.md"],
    "codes/": {
        "python/": [
          "python_1.md",
          "python_2.md",
          "python_3.md",
          "python_4.md",
          "python_5.md",
          "python_6.md",
          "python_7.md",
          "python_8.md",
          "python_9.md",
          "python_10.md"
        ]
    },
    "linux/": {
        "arch/": [
          "arch_1.md", 
          "arch_2.md", 
          "arch_3.md", 
          "arch_4.md", 
          "arch_5.md", 
          "arch_6.md", 
          "arch_7.md", 
          "arch_8.md",
          "arch_9.md"
        ]
    },
    "about/": ["about.md"]
};

const quoteMap = [
    {
        text: "Those who would give up essential liberty, to purchase a little temporary safety, deserve neither liberty nor safety.",
        author: "Benjamin Franklin",
        reference: "Pennsylvania Assembly: Reply to the Governor, November 11, 1755"
    },
    {
        text: "It is better that ten guilty persons escape than that one innocent suffer.",
        author: "Sir William Blackstone",
        reference: "Commentaries on the Laws of England, 1765"
    },
    {
        text: "If all mankind minus one, were of one opinion, and only one person were of the contrary opinion, mankind would be no more justified in silencing that one person, than he, if he had the power, would be justified in silencing mankind.",
        author: "John Stuart Mill",
        reference: "On Liberty, 1859"
    },
    {
        text: "Power tends to corrupt and absolute power corrupts absolutely.",
        author: "Lord Acton",
        reference: "Letter to Bishop Mandell Creighton, April 5, 1887"
    },
    {
        text: "Is man only a blunder of God? Or is God only a blunder of man?",
        author: "Friedrich Wilhelm Nietzsche",
        reference: "Twilight of the Idols, or, How to Philosophize with a Hammer, 1889"
    },
    {
        text: "That which does not kill me, makes me stronger.",
        author: "Friedrich Wilhelm Nietzsche",
        reference: "Twilight of the Idols, or, How to Philosophize with a Hammer, 1889"
    },
    {
        text: "I disapprove of what you say, but I will defend to the death your right to say it.",
        author: "Evelyn Beatrice Hall",
        reference: "The Friends of Voltaire, 1906"
    },
    {
        text: "A government big enough to give you everything you want is big enough to take away everything you have.",
        author: "Gerald Rudolph Ford Jr.",
        reference: "Address to a Joint Session of Congress, August 12, 1974"
    }
];

//  ################
//  HELPER FUNCTIONS
//  ################

function getBaseName(path) {
    const baseName = path.split('/').pop().replace(/\..*$/, '');
    return baseName.replace(/_([a-z])/g, (_, letter) => letter.toUpperCase());
}

function getRandomPath(obj, path) {
    if (Array.isArray(obj)) return path + randomizer(obj);
    const keys = Object.keys(obj);
    if (keys.length === 0) return path;
    const key = randomizer(keys);
    const value = obj[key];
    return getRandomPath(value, path + key);
}

function getAllPaths(obj, path, exclude = []) {
    let results = [];
    if (Array.isArray(obj)) {
        results.push(obj.map((file) => path + file));
    } else if (typeof obj === "object" && obj !== null) {
        for (const key in obj) {
            if (exclude.includes(key)) continue;
            const value = obj[key];
            const newPath = path + key;
            results = results.concat(getAllPaths(value, newPath, exclude));
        }
    }
    return results;
}

function randomizer(array) {
    const randomIndex = Math.floor(Math.random() * array.length);
    return array[randomIndex];
}

function parseAndPurify(markdown) {
    const markdownObject = marked.parse(markdown);
    return DOMPurify.sanitize(markdownObject);
}

function headingToken(markdown) {
    const tokens = marked.lexer(markdown);
    return [tokens.find((token) => token.type === "heading")];
    //return tokens.filter((token) => token.type === "heading");  #  Alternative use that returns an array of all 'headings'
}

function toCamelCase(string) {
    return string.
    toLowerCase()
    .split(' ')
    .map((word, index) => {
        if (index === 0) {
            return word;
        } else {
            return word.charAt(0).toUpperCase() + word.slice(1);
        }
    })
    .join('');
}

//  -------------------------------------------------------------------------------------------
//  REFERENCE: https://www.w3resource.com/javascript-exercises/javascript-math-exercise-105.php
//  -------------------------------------------------------------------------------------------
function numberToWord(number) {
    if (number < 0)
        return false;
    const singleDigit = ['', "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"];
    const doubleDigit = ["Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen"];
    const belowHundred = ["Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"];
    if (number === 0) return "Zero";
    function translate(number) {
        let word = '';
        if (number < 10) {
            word = singleDigit[number];
        } else if (number < 20) {
            word = doubleDigit[number - 10];
        } else if (number < 100) {
            let rem = translate(number % 10);
            word = belowHundred[(number - number % 10) / 10 - 2] + rem;
        } else if (number < 1000) {
            word = singleDigit[Math.trunc(number / 100)] + "Hundred" + translate(number % 100);
        } else if (number < 1000000) {
            word = translate(parseInt(number / 1000)).trim() + "Thousand" + translate(number % 1000);
        } else if (number < 1000000000) {
            word = translate(parseInt(number / 1000000)).trim() + "Million" + translate(number % 1000000);
        } else {
            word = translate(parseInt(number / 1000000000)).trim() + "Billion" + translate(number % 1000000000);
        }
        return word;
    }
    let result = translate(number);
    return result.trim();
}

function preTitle(string) {
    const base = getBaseName(string).split('_');
    if (base[1] === undefined) {
        return base[0];
    }
    return base[0] + numberToWord(base[1]);
}

//  ###############
//  CONTENT MODULES
//  ###############

async function createFooterContent() {    
    const year = new Date().getFullYear();
    const htmlContent = `
    <p class="footer-text">
        &copy; ${year} All rights reserved.<br>Powered by 
        <a href="https://en.wikipedia.org/wiki/HTML" class="icon" title="HTML">
            <img class="icon" src="assets/images/html_32dp_000000_FILL0_wght700_GRAD200_opsz48.svg" alt="HTML">
        </a>
        <a href="https://en.wikipedia.org/wiki/JavaScript" class="icon" title="JavaScript">
            <img class="icon" src="assets/images/javascript_32dp_000000_FILL0_wght700_GRAD200_opsz48.svg" alt="JavaScript">
        </a>
        <a href="https://en.wikipedia.org/wiki/CSS" class="icon" title="CSS">
            <img class="icon" src="assets/images/css_32dp_000000_FILL0_wght700_GRAD200_opsz48.svg" alt="CSS">
        </a>and designed by buburakk.
    </p>
    `;
    return htmlContent;
}

async function createLogoContent() {
    const htmlContent = `
    <div data-navigation="assets/index/index.md" id="buburakkLogo" title="Logo">
        <span class="logo-text bu-part">bu</span>
        <span class="logo-text burak-part">burak</span>
        <span class="logo-text k-part">k</span>
    </div>
    `;
    return htmlContent;
}

async function createImageContent() {
    const htmlContent = `
    <div class="image-frame">
        <img src="assets/images/walk.gif" alt="Walk">
    </div>
    `;
    return htmlContent;
}

async function createCardContent() {
    const htmlContent = `
    <div class="image-frame">
        <img src="assets/images/me.jpg" alt="Me">
        <div class="image-subframe">
        <a href="https://www.linkedin.com/in/b-burak-karaosmano%C4%9Flu-951a00178/">
            <img src="assets/images/linkedin_icon.svg" alt="LinkedIn">
        </a>    
        <a href="https://github.com/buburakk">
            <img src="assets/images/github_icon.svg" alt="GitHub">
        </a>
        </div>
    </div>
    `;
    return htmlContent;
}

async function createDropDownMenu(title, markdowns) {
    let htmlContent = `
    <li class="dropdown-element">
        <a data-navigation="${randomizer(markdowns)}">
            <span>${title}</span>
        </a>
    `;
    for (const markdown of markdowns) {
        try {
            const response = await fetch(markdown);
            if (!response.ok) {
                throw new Error("Failed to load 'Markdown' (.md) file!");
            }
            const headings = headingToken(await response.text());
            htmlContent += "<ul>";
            for (const heading of headings) {
                htmlContent += `
                <li>
                    <a data-navigation="${markdown}">
                        <span>${heading.text}</span>
                    </a>
                </li>
                `;
            }
            htmlContent += "</ul>";
        } catch (error) {
            console.error(error);
        }
    }
    htmlContent += "</li>";
    return htmlContent;
}

async function createMarkdownContent(title, markdowns) {
    const articleName = toCamelCase(title);
    let htmlContent = `
    <article id="${articleName + "Article"}">
    `;
    for (const markdown of markdowns) {
        try {
            const response = await fetch(markdown);
            if (!response.ok) {
                throw new Error("Failed to load 'Markdown' (.md) file!");
            }
            const sectionName = preTitle(markdown);
            htmlContent += `
            <section id="${sectionName + "Section"}">
                ${parseAndPurify(await response.text())}
            </section>
            `;
        } catch (error) {
            console.error(error);
        }
    }
    htmlContent += "</article>";
    return htmlContent;
}

async function createQuoteContent(quotes) {
    try {
        const quote = randomizer(quotes);
        const htmlContent = `
        <blockquote>
            <p class="quote-text">
                ${quote.text}
            </p>
            <p class="quote-author-reference">
                — ${quote.author}<br>(${quote.reference})
            </p>
        </blockquote>
        `;
        return htmlContent;
    } catch (error) {
        console.error(error);
    }
}

//  ##############
//  LOADER MODULES
//  ##############

async function loadLogoContent() {
    const element = document.querySelector("div.header-logo-container");
    element.innerHTML = await createLogoContent();
}

async function loadHomeImage() {
    const element = document.querySelector("#indexSection h1");
    element.insertAdjacentHTML("beforebegin", await createImageContent());
}

async function loadCardContent() {
    const element = document.querySelector("#aboutSection h1");
    element.insertAdjacentHTML("beforebegin", await createCardContent());
}

async function loadDropDownMenu(titles, markdowns) {
    const element = document.querySelector("div.dropdown-container");
    let htmlContent = `
    <nav>
        <ul class="dropdown-menu-elements">
    `;
    for (let i = 0; i < Math.min(titles.length, markdowns.length); i++) {
        htmlContent += await createDropDownMenu(titles[i], markdowns[i]);
    }
    htmlContent += "</ul>";
    htmlContent += "</nav>";
    element.innerHTML = htmlContent;
}

async function loadMarkdownContent(titles, markdowns) {
    const element = document.querySelector("div.article-container");
    let htmlContent = '';
    for (let i = 0; i < Math.min(titles.length, markdowns.length); i++) {
        htmlContent += await createMarkdownContent(titles[i], [markdowns[i]]);
    }
    element.innerHTML = htmlContent;
    const codeBlocks = document.querySelectorAll("pre code");
    async function highlightAllCodes() {
        for (let code of codeBlocks) {
            if (code.hasAttribute("class") && code.classList.length > 0) {
                await loadHighlightJS(code);
            } else {
                code.style.display = "block";
                code.style.margin = "0.5rem";
                code.style.border = "var(--cd-brdr-styl)";
                code.style.borderRadius = "var(--cd-brdr-rds)";
                code.style.padding = "0.5rem";
                code.style.backgroundColor = "rgba(231,231,231,1)";
                code.style.overflowX = "auto";
            }
        }
    }
    await highlightAllCodes();
    await loadMathJax(element);
}

async function loadQuoteContent() {
    const element = document.querySelector("div.quote-container");
    element.innerHTML = await createQuoteContent(quoteMap);
}

async function loadFooterContent() {
    const element = document.querySelector("div.footer-container");
    element.innerHTML = await createFooterContent();
}

function loadAboutLayout() {
    window.scrollTo(0, 0);
    const asideElement = document.querySelector("aside");
    if (asideElement) asideElement.remove();
    const mainElement = document.querySelector("main");
    //mainElement.style.marginRight = "12.5%";
    mainElement.style.marginLeft = "12.5%";
    mainElement.style.width = "75%";
    handleMediaQuery();
}

const defaultAsideElement = document.querySelector("aside").cloneNode(true);

function loadDefaultLayout() {
    window.scrollTo(0, 0);
    const mainElement = document.querySelector("main");
    let asideElement = document.querySelector("aside");
    //mainElement.style.margin = '';
    if (!asideElement) {
        const headerElement = document.querySelector("header");
        headerElement.insertAdjacentElement("afterend", defaultAsideElement);
        asideElement = document.querySelector("aside");
    }
    /*mainElement.style.width = `
    calc(100% - ${window.getComputedStyle(asideElement).width} 
    - ${window.getComputedStyle(asideElement).marginRight} 
    - ${window.getComputedStyle(mainElement).marginLeft})
    `;
    mainElement.style.width = '';*/
    handleMediaQuery();
}

//  ################
//  MathJax RENDERER
//  ################

let isMathJaxLoaded = null;

async function loadMathJax(element) {
    if (isMathJaxLoaded) {
        if (window.MathJax && MathJax.typesetPromise) {
            if (element) await MathJax.typesetPromise([element]);
            return;
        } else {
            throw new Error("'MathJax' is not ready after loading!");
        }
    }
    window.MathJax = {
        tex: {
            inlineMath: [['$', '$'], ["\\(", "\\)"]],
            displayMath: [["$$", "$$"], ["\\[", "\\]"]]/*,
            packages: {"[+]": ["ams"]}
        },
        svg: {fontCache: "global"*/}
    };
    isMathJaxLoaded = new Promise((resolve, reject) => {
        const script = document.createElement("script");
        script.src = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js";
        script.async = true;
        script.onload = async () => {
            try {
                if (window.MathJax && MathJax.typesetPromise) {
                    if (element) await MathJax.typesetPromise([element]);
                    resolve();
                } else {
                    reject(new Error("'MathJax' failed to initialize!"));
                }
            } catch (error) {
                reject(error);
            }
        };
        script.onerror = () => reject(new Error("Failed to load 'MathJax' script!"));
        document.head.appendChild(script);
    });
    return isMathJaxLoaded;
}

//  ################
//  CODE HIGHLIGHTER
//  ################

let isHighlightJSLoaded = null;

async function loadHighlightJS(element) {
    if (isHighlightJSLoaded) {
        if (window.hljs) {
            if (element) {
                hljs.highlightElement(element);
            } else {
                hljs.highlightAll();
            }
            return;
        } else {
            throw new Error("'highlight.js' is not ready after loading!");
        }
    }
    isHighlightJSLoaded = new Promise((resolve, reject) => {
        const link = document.createElement("link");
        link.rel = "stylesheet";
        link.href = "https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.11.1/build/styles/github-dark.min.css";
        link.onload = () => {};
        link.onerror = () => reject(new Error("Failed to load highlight.js CSS!"));
        document.head.appendChild(link);
        const script = document.createElement("script");
        script.src = "https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.11.1/build/highlight.min.js";
        script.async = true;
        script.onload = () => {
            try {
                if (window.hljs) {
                    hljs.configure({ ignoreUnescapedHTML: true });
                    if (element) {
                        hljs.highlightElement(element);
                    } else {
                        hljs.highlightAll();
                    }
                    resolve();
                } else {
                    reject(new Error("'highlight.js' failed to initialize!"));
                }
            } catch (error) {
                reject(error);
            }
        };
        script.onerror = () => reject(new Error("Failed to load 'highlight.js' script!"));
        document.head.appendChild(script);
    });
    return isHighlightJSLoaded;
}

//  ###############################
//  TOGGLE NAVIGATION BAR ANIMATION
//  ###############################

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
    //navbarVisible ? showNavbar() : hideNavbar();  #  Alternative use of the condition below
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

function enableNavbar() {
    disableNavbar();
    burgerMenuElement.addEventListener("click", onBurgerClick);
    document.addEventListener("click", onDocumentClick);
    hideNavbar();
}

function disableNavbar() {
    burgerMenuElement.removeEventListener("click", onBurgerClick);
    document.removeEventListener("click", onDocumentClick);
    toggleMenuElements.style.display = '';
    navbarVisible = false;
}

//  #######################
//  RESPONSIVE PAGE HANDLER
//  #######################

function handleMediaQuery() {
    const mainElement = document.querySelector("main");
    const mainContainerElement = document.querySelector("div.main-container");
    const asideElement = document.querySelector("aside");
    const windowWidth = window.innerWidth;
    if (windowWidth <= 704) {
        enableNavbar();
        mainElement.style.marginLeft = '';
        mainElement.style.marginRight = '';
        mainElement.style.width = '';
        mainElement.style.paddingRight = '0';
        mainElement.style.paddingLeft = '';
        mainContainerElement.style.borderRight = "none";
        mainContainerElement.style.borderTopRightRadius = '0';
        mainContainerElement.style.borderBottomRightRadius = '0';

        mainContainerElement.style.borderLeft = "5px solid black";
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
            mainElement.style.marginLeft = "15%";
            mainElement.style.width = "70%";
            mainElement.style.paddingLeft = "3rem";
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
