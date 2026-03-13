// ###############
// CONTENT MODULES
// ###############

import { randomizer, getMarkdown, toCamelCase, preTitle, parseAndPurify } from "./utils.js";

export const quoteMap = [
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

export async function createFooterContent() {    
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
        </a>and designed by
        <a href="https://www.linkedin.com/in/b-burak-karaosmano%C4%9Flu-951a00178/" style="padding-top: 4px;">
            buburakk
        </a>.
    </p>
    `;
    return htmlContent;
}

export async function createLogoContent() {
    const htmlContent = `
    <div data-navigation="index/index.md" id="buburakkLogo" title="Logo">
        <span class="logo-text bu-part">bu</span>
        <span class="logo-text burak-part">burak</span>
        <span class="logo-text k-part">k</span>
    </div>
    `;
    return htmlContent;
}

export async function createImageContent() {
    const htmlContent = `
    <div class="image-frame">
        <img src="assets/images/walk.gif" alt="Walk">
    </div>
    `;
    return htmlContent;
}

export async function createCardContent() {
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

export async function createDropdownMenu(title, markdowns) {
    // data-navigation="${randomizer(markdowns).path}" This can be added to 'a' element to impart it hyperlink property
    let htmlContent = `
    <li class="dropdown-element">
        <a>
            <span>${title}</span>
        </a>
        <ul>
    `;
    for (const item of markdowns) {
        htmlContent += `
        <li>
            <a data-navigation="${item.path}">
                <span>${item.title}</span>
            </a>
        </li>
        `;
    }
    htmlContent += "</ul></li>";
    return htmlContent;
}

export async function createMarkdownContent(title, markdowns) {
    const articleName = toCamelCase(title);
    let htmlContent = `
    <article id="${articleName + "Article"}">
    `;
    for (const markdown of markdowns) {
        try {
            // const response = await fetch(markdown);
            // if (!response.ok) {
            //     throw new Error("Failed to load 'markdown' (.md) file!");
            // }
            const text = await getMarkdown(markdown);
            const sectionName = preTitle(markdown);
            htmlContent += `
            <section id="${sectionName + "Section"}">
                ${parseAndPurify(text)}
            </section>
            `;
        } catch (error) {
            console.error(error);
        }
    }
    htmlContent += "</article>";
    return htmlContent;
}

export async function createQuoteContent(quotes) {
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
