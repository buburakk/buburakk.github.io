# How to Customize "fastfetch"

After setting up a Linux distro, running `neofetch` command has been presumably the first thing to do by users. It is a commonly used, or used to be so, package that retrieves system information, prints them on command-line interface. It is no longer actively maintained so it is better to switch to an alternative, which is `fastfetch`. First, install **fastfetch** package without *sudo* (if you logged in as **root**): 

```bash
pacman -S fastfetch
```

Create a hidden **.config** directory in your home directory (assuming a user named **archie** logged in):

```bash
mkdir ~/.config
```

Create a configuration file (where **\~/.config/fastfetch/config.jsonc**):

```bash
fastfetch --gen-config
```

Just remove or comment out unwanted parts:

```bash
// "disk",
// "localip",
// "battery",
// "poweradapter",
```
