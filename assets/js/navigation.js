// ##################
// NAVIGATION HANDLER
// This is the final version of 'navigation' script handles link to page in URL bar.
// ##################

import { handleMediaQuery } from "./responsive.js";
import { loadAboutLayout, loadDefaultLayout, loadLogoContent, 
          loadHomeImage, loadCardContent, loadDropdownMenu, 
            loadMarkdownContent, loadQuoteContent, loadFooterContent } from "./loaders.js";
import { getAllPaths, getRandomPath, updateURL } from "./utils.js";

const navigationMap = {
    "index/": [{ path: "index.md", title: "Welcome!" }],
    "notes/": [{ path: "update.md", title: "Updating..." }],
    "codes/": {
        "python/": [
            { path: "python_1.md", title: "Benchmarking Prime Number Generation: Naive Division vs. Optimized Methods" },
            { path: "python_2.md", title: "Animation in Python with Matplotlib" },
            { path: "python_3.md", title: "Cartesian Products in NumPy" },
            { path: "python_4.md", title: "Automatic Differentiation on 2D Domains with JAX" },
            { path: "python_5.md", title: "Time Series Regression with PyTorch" },
            { path: "python_6.md", title: "Creating Synthetic Time Series from Interpretable Components" },
            { path: "python_7.md", title: "Efficient Multivariate Taylor Expansions with JAX" },
            { path: "python_8.md", title: "Computing Pascal's Triangle Efficiently" },
            { path: "python_9.md", title: "Linear Systems and Finite Difference Stencils in NumPy" },
            { path: "python_10.md", title: "Finite Difference Differentiation: Accuracy and Error" },
            { path: "python_11.md", title: "Array Compression: A Problem from Engineering" }
        ]
    },
    "linux/": {
        "arch/": [
            { path: "arch_1.md", title: "Coloring Terminal Outputs with \"dircolors\" Package" },
            { path: "arch_2.md", title: "File Format Error When Running" },
            { path: "arch_3.md", title: "How to Customize \"fastfetch\"" },
            { path: "arch_4.md", title: "How to Fix \"visudo: no editor found\" Error" },
            { path: "arch_5.md", title: "How to Remove a Package Safely by \"pacman\"" },
            { path: "arch_6.md", title: "How to Set Up \"sudo\"" },
            { path: "arch_7.md", title: "Updating File Content" },
            { path: "arch_8.md", title: "Updating \"man\" Database" },
            { path: "arch_9.md", title: "Searching for a File in the Repository" },
            { path: "arch_10.md", title: "File Conversion: Pandoc 'em All!" }
        ]
    },
    "about/": [{ path: "about.md", title: "Who Am I?" }]
};

function prepareLinks() {
  const notesButtons = document.querySelectorAll('[id*="notesButton"]');
  notesButtons.forEach((element) => element.dataset.navigation = getRandomPath(navigationMap["notes/"], "notes/"));
  const codesButtons = document.querySelectorAll('[id*="codesButton"]');
  codesButtons.forEach((element) => element.dataset.navigation = getRandomPath(navigationMap["codes/"], "codes/"));
  const linuxButtons = document.querySelectorAll('[id*="linuxButton"]');
  linuxButtons.forEach((element) => element.dataset.navigation = getRandomPath(navigationMap["linux/"], "linux/"));
}

async function handleRoute(base, selection) {
  if (selection.startsWith("index/")) {
    loadDefaultLayout();
    handleMediaQuery();
    await loadDropdownMenu(["Python", "Arch Linux"], getAllPaths(navigationMap, '', ["index/", "notes/", "about/"]));
    await loadMarkdownContent(["index"], [`${base}index/index.md`]);
    await loadHomeImage();
    await loadQuoteContent();
    return;
  }
  if (selection.startsWith("about/")) {
    loadAboutLayout();
    handleMediaQuery();
    await loadMarkdownContent(["about"], [`${base}about/about.md`]);
    await loadCardContent();
    await loadQuoteContent();
    return;
  }
  loadDefaultLayout();
  handleMediaQuery();
  window.scrollTo(0,0);
  const navigationNodes = selection.split('/');
  const title = navigationNodes[1];
  if (selection.startsWith("notes/")) {
    await loadDropdownMenu(["Updating..."], getAllPaths(navigationMap["notes/"], "notes/"));
  }
  else if (selection.startsWith("codes/")) {
    await loadDropdownMenu(["Python"], getAllPaths(navigationMap["codes/"], "codes/"));
  }
  else if (selection.startsWith("linux/")) {
    await loadDropdownMenu(["Arch Linux"], getAllPaths(navigationMap["linux/"], "linux/"));
  }
  await loadMarkdownContent([title], [`${base}${selection}`]);
  await loadQuoteContent();
}

export async function handleNavigation() {
  const parts = window.location.pathname.split('/');
  let BASE_PATH;
  if (parts[1] !== '') {
    BASE_PATH = '/' + parts[1] + '/';
  } else {
    BASE_PATH = '/';
  }
  handleMediaQuery();
  window.addEventListener("resize", handleMediaQuery);
  document.addEventListener("click", async (event) => {
    const button = event.target.closest("div[data-navigation], a[data-navigation]");
    if (!button) return;
    const selection = button.dataset.navigation;
    // console.log(selection); Uncomment this for testing
    if (selection === "scrolltop") {
      window.scrollTo(0,0);
      return;
    }
    updateURL(BASE_PATH, selection, false);
    await handleRoute(BASE_PATH, selection);
  });
  window.addEventListener("popstate", async () => {
    const params = new URLSearchParams(window.location.search);
    const path = params.get("page");
    if (path) {
      await handleRoute(BASE_PATH, path);
    }
  });
  await loadLogoContent();
  prepareLinks();
  await loadDropdownMenu(["Python", "Arch Linux"], getAllPaths(navigationMap, '', ["index/", "notes/", "about/"]));
  const params = new URLSearchParams(window.location.search);
  const initialPath = params.get("page") || "index/index.md";
  updateURL(BASE_PATH, initialPath, true);
  await handleRoute(BASE_PATH, initialPath);
  await loadFooterContent();
}
