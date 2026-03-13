// #########
// UTILITIES
// #########

export function getBaseName(path) {
    const baseName = path.split('/').pop().replace(/\..*$/, '');
    return baseName.replace(/_([a-z])/g, (_, letter) => letter.toUpperCase());
}

export function getRandomPath(obj, path) {
    if (Array.isArray(obj)) {
        const item = randomizer(obj);
        return path + item.path;
    }
    const keys = Object.keys(obj);
    if (keys.length === 0) return path;
    const key = randomizer(keys);
    const value = obj[key];
    return getRandomPath(value, path + key);
}

export function getAllPaths(obj, path, exclude = []) {
    let results = [];
    if (Array.isArray(obj)) {
        results.push(obj.map((item) => ({ 
            path: path + item.path, 
            title: item.title 
        })));
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

const markdownCache = new Map();

export async function getMarkdown(path) {
    if (markdownCache.has(path)) {
        return markdownCache.get(path);
    }
    const promise = await fetch(path)
        .then((response) => {
            if (!response.ok) throw new Error(`Failed to load 'markdown' (.md) file: ${path}`);
            return response.text();
        })
        .then((text) => {
            markdownCache.set(path, Promise.resolve(text));
            return text;
        })
        .catch((error) => {
            markdownCache.delete(path);
            throw error;
        });
    markdownCache.set(path, promise);
    return promise;
}

// const imageCache = new Map();

// export function getImage(path) {
//     if (imageCache.has(path)) {
//         return imageCache.get(path);
//     }
//     const promise = new Promise((resolve, reject) => {
//         const img = new Image();
//         img.src = path;
//         img.onload = () => {
//             resolve(img);
//         };
//         img.onerror = () => {
//             imageCache.delete(path);
//             reject(new Error(`Failed to load image: ${path}`));
//         };
//     });
//     imageCache.set(path, promise);
//     return promise;
// }

export function updateURL(base, path, replace) {
    const url = `${base}?page=${encodeURIComponent(path)}`;
    if (replace) {
        history.replaceState({path}, '', url);
    } else {
        history.pushState({path}, '', url);
    }
}

export function randomizer(array) {
    const randomIndex = Math.floor(Math.random() * array.length);
    return array[randomIndex];
}

const renderer = {
    code(obj) {
        // console.log(obj); Uncomment this for testing
        const lang = obj.lang || "plaintext";
        const highlighted = window.hljs.highlight(obj.text, { language: lang }).value;
        const lines = highlighted
            .split("\n")
            .map((line, index) => `<span class="code-line" data-line="${numberToWord(index + 1)}">${line}</span>`)
            .join("\n");
        // console.log(lines); Uncomment this for testing
        return `
        <details class="code-toggle">
            <summary></summary>
            <pre>
                <code class="language-${lang} hljs" data-highlighted="yes">${lines}</code>
            </pre>
        </details>
        `;
    }
};

marked.use({ renderer });

export function parseAndPurify(markdown) {
    const markdownObject = marked.parse(markdown);
    return DOMPurify.sanitize(markdownObject);
}

// function headingToken(markdown) {
//     const tokens = marked.lexer(markdown);
//     return [tokens.find((token) => token.type === "heading")];
//     //return tokens.filter((token) => token.type === "heading");  #  Alternative use that returns an array of all 'headings'
// }

export function toCamelCase(string) {
    return string
        .toLowerCase()
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

export function containsMath(element) {
    const text = element.textContent;
    return /(\$[^$]+\$)|(\$\$[^$]+\$\$)|(\\\(.+?\\\))|(\\\[.+?\\\])/.test(text) || element.querySelector("math");
}

// -------------------------------------------------------------------------------------------
// REFERENCE: https://www.w3resource.com/javascript-exercises/javascript-math-exercise-105.php
// -------------------------------------------------------------------------------------------
export function numberToWord(number) {
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

export function preTitle(string) {
    const base = getBaseName(string).split('_');
    if (base[1] === undefined) {
        return base[0];
    }
    return base[0] + numberToWord(base[1]);
}

// ################
// MathJax RENDERER 
// NOTE: Include folders/files inside 'mathjax' folder: 'input', 'output', 'sre', 'ui', 
// 'tex-chtml.js', 'tex-mml-chtml.js', 'tex-svg.js'
// ################

window.MathJax = {
    loader: {
        load: ["input/tex", "output/chtml"]
    },
    tex: {
        inlineMath: [['$', '$'], ["\\(", "\\)"]],
        displayMath: [["$$", "$$"], ["\\[", "\\]"]]
    },
    output: {
        displayOverflow: "linebreak",
        linebreaks: {
            inline: true,
            width: "100%",
            lineleading: 0.25
        }
    }
};

export async function renderMathJax(element) {
    if (containsMath(element)) {
        await MathJax.startup.pageReady;
        await MathJax.typesetPromise([element]);
    }
}

// ################
// CODE HIGHLIGHTER
// ################

window.hljs.configure({ ignoreUnescapedHTML: true });

// export async function highlightCodes() {
//     const codeBlocks = document.querySelectorAll("pre code");
//     if (codeBlocks.length > 0) {
//         for (const code of codeBlocks) {
//             if (code.hasAttribute("class") && code.classList.length > 0) {
//                 window.hljs.highlightElement(code)
//             } else {
//                 code.style.display = "block";
//                 code.style.margin = "0.5rem";
//                 code.style.border = "var(--cd-brdr-styl)";
//                 code.style.borderRadius = "var(--cd-brdr-rds)";
//                 code.style.padding = "0.5rem";
//                 code.style.backgroundColor = "rgba(231,231,231,1)";
//                 code.style.overflowX = "auto";
//             }
//         }
//     }
// }
