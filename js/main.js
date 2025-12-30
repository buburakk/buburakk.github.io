(async () => {
  handleMediaQuery();
  window.addEventListener("resize", handleMediaQuery);
  document.addEventListener("click", async (event) => {
    const button = event.target.closest("div[data-navigation], a[data-navigation]");
    if (!button) return;
    const selection = button.dataset.navigation;
    if (selection.includes("index/")) {
      loadDefaultLayout();
      await loadDropDownMenu(["Python", "Arch Linux"], getAllPaths(navigationMap, "assets/", ["index/", "notes/", "about/"]));
      await loadMarkdownContent(["index"], [selection]);
      await loadHomeImage();
      await loadQuoteContent();
    } else if (!selection.includes(".md") && selection.includes("notes/")) {
      loadDefaultLayout();
      await loadDropDownMenu(["Updating..."], getAllPaths(navigationMap["notes/"], "assets/notes/"));
      await loadMarkdownContent(["update"], ["assets/notes/update.md"]);
      await loadQuoteContent();
    } else if (!selection.includes(".md") && selection.includes("codes/")) {
      loadDefaultLayout();
      await loadDropDownMenu(["Python"], getAllPaths(navigationMap["codes/"], "assets/codes/"));
      await loadMarkdownContent(["codes"], [getRandomPath(navigationMap["codes/"], "assets/codes/")]);
      await loadQuoteContent();
    } else if (!selection.includes(".md") && selection.includes("linux/")) {
      loadDefaultLayout();
      await loadDropDownMenu(["Arch Linux"], getAllPaths(navigationMap["linux/"], "assets/linux/"));
      await loadMarkdownContent(["linux"], [getRandomPath(navigationMap["linux/"], "assets/linux/")]);
      await loadQuoteContent();
    } else if (selection.includes("about/")) {
      loadAboutLayout();
      await loadMarkdownContent(["about"], [selection]);
      await loadCardContent();
      await loadQuoteContent();
    } else if (selection.includes("scrolltop")) {
      window.scrollTo(0, 0);
    } else if (document.querySelector("div.dropdown-container")) {
      window.scrollTo(0, 0);
      const navigationNodes = selection.split('/');
      const title = navigationNodes[1];
      await loadMarkdownContent([title], [selection]);
      await loadQuoteContent();
    }
  });
  await loadLogoContent();
  await loadDropDownMenu(["Python", "Arch Linux"], getAllPaths(navigationMap, "assets/", ["index/", "notes/", "about/"]));
  await loadMarkdownContent(["index"], ["assets/index/index.md"]);
  await loadHomeImage();
  await loadQuoteContent();  
  await loadFooterContent();
})();
