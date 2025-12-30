# How to Remove a Package Safely by "pacman"

Imagine there is a package **inspac** installed on our machine. And at some point we want to safely remove it. The information of package in question should first be viewed by running:

```bash
pacman -Qi inspac
```

The lines that should be checked for start with titles **Provides**, **Depends On**, **Optional Deps**, **Required By**, **Optional For**, **Conflicts With**, **Replaces** before removing the package with related **config** files and unused dependencies:

```bash
sudo pacman -Rns inspac
```

Then we can also check for missing files that might have been required:

```bash
sudo pacman -Qk | grep "missing"
```
