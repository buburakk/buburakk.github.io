# Searching for a File in the Repository

There might be cases where a specific file may be required for a package/program to run. There is a easy way that helps with locating it in the repository, inside which package(s). If command has not been used before, a database must be set up first by running the command:

```bash
sudo pacman -Fy
```

Next, a search across the repository for the file, say **cancel.sty**, can be implemented by the following:

```bash
pacman -F cancel.sty
```

At the end of the search, it returns an output similar to the one below:

```bash
extra/texlive-mathscience 2025.2-1
    usr/share/texmf-dist/tex/latex/cancel/cancel.sty
```

The package in question is located within **texlive-mathscience** package inside **usr/share/texmf-dist/tex/latex/cancel/** directory.
