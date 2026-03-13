# File Conversion: Pandoc 'em All!

**pandoc** package is literally all-purpose tool to convert most, if not all, types of file from one to another. If you have a **.tex** file that you want to convert it into **.html** format for your blog post. If it has not been installed, then we install the package and its dependencies:

```bash
sudo pacman -S pandoc-cli
```

In the next step, we run the command below to convert our file:

```bash
pandoc file_name.tex -s -o file_name.html --mathjax
```

The command option `--mathjax` automatically handles the libraries that you used in the preamble of your **.tex** document.
