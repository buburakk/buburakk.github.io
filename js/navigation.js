//  ##########
//  NAVIGATION
//  ##########

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
          "arch_9.md",
          "arch_10.md"
        ]
    },
    "about/": ["about.md"]
};

(async () => {
  handleMediaQuery();
  window.addEventListener("resize", handleMediaQuery);
  const dropdownContainer = document.querySelector("div.dropdown-container");
  const dropdownPaths = {
    default: getAllPaths(navigationMap, "assets/", ["index/", "notes/", "about/"]),
    notes: getAllPaths(navigationMap["notes/"], "assets/notes/"),
    codes: getAllPaths(navigationMap["codes/"], "assets/codes/"),
    linux: getAllPaths(navigationMap["linux/"], "assets/linux/")
  };
  const allMarkdownsToPreload = [
    "assets/index/index.md",
    ...dropdownPaths.default.flat(),
    ...dropdownPaths.notes.flat(),
    ...dropdownPaths.codes.flat(),
    ...dropdownPaths.linux.flat()
  ];
  await Promise.allSettled(allMarkdownsToPreload.map((path) => Promise.all([
    fetchMarkdown(path),
    fetchHeadings(path)
  ]).catch(console.error)));
  document.addEventListener("click", async (event) => {
    const button = event.target.closest("div[data-navigation], a[data-navigation]");
    if (!button) return;
    const selection = button.dataset.navigation;
    const hasMarkdown = selection.includes(".md");
    if (selection.includes("index/")) {
      loadDefaultLayout();
      await loadDropDownMenu(["Python", "Arch Linux"], dropdownPaths.default);
      await loadMarkdownContent(["index"], [selection]);
      await loadHomeImage();
      await loadQuoteContent();
    } else if (!hasMarkdown && selection.includes("notes/")) {
      loadDefaultLayout();
      await loadDropDownMenu(["Updating..."], dropdownPaths.notes);
      await loadMarkdownContent(["update"], ["assets/notes/update.md"]);
      await loadQuoteContent();
    } else if (!hasMarkdown && selection.includes("codes/")) {
      loadDefaultLayout();
      await loadDropDownMenu(["Python"], dropdownPaths.codes);
      await loadMarkdownContent(["codes"], [getRandomPath(navigationMap["codes/"], "assets/codes/")]);
      await loadQuoteContent();
    } else if (!hasMarkdown && selection.includes("linux/")) {
      loadDefaultLayout();
      await loadDropDownMenu(["Arch Linux"], dropdownPaths.linux);
      await loadMarkdownContent(["linux"], [getRandomPath(navigationMap["linux/"], "assets/linux/")]);
      await loadQuoteContent();
    } else if (selection.includes("about/")) {
      loadAboutLayout();
      await loadMarkdownContent(["about"], [selection]);
      await loadCardContent();
      await loadQuoteContent();
    } else if (selection.includes("scrolltop")) {
      window.scrollTo(0, 0);
    } else if (dropdownContainer) {
      window.scrollTo(0, 0);
      const navigationNodes = selection.split('/');
      const title = navigationNodes[1];
      await loadMarkdownContent([title], [selection]);
      await loadQuoteContent();
    }
  });
  await loadLogoContent();
  await loadDropDownMenu(["Python", "Arch Linux"], dropdownPaths.default);
  await loadMarkdownContent(["index"], ["assets/index/index.md"]);
  await loadHomeImage();
  await loadQuoteContent();
  await loadFooterContent();
})();
