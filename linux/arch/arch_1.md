# Coloring Terminal Outputs with "dircolors" Package

Terminal output colors of `ls` command are by default in plain colors. However, this can be changed by **dircolors** package. Run this command in order to create a default reference file named **config** for coloring assuming **\~/.config/dircolors** path already exists:

```bash
dircolors -p > ~/.config/dircolors/config
```

In order to make the change permanent, inside **\~/.bashrc** file we add:

```bash
eval "$(dircolors -b ~/.config/dircolors/config)"
```

Finally, running the command below makes the change take effect:

```bash
source ~/.bashrc
```
