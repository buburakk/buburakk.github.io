# How to Set Up "sudo"

First install **sudo** package (since you are logged in as root): 

```bash
pacman -S sudo
```

Create a user (if there is not): 

```bash
useradd -m archie
```

Then modify the user character: 

```bash
usermod -aG wheel archie
```

Safely editing **/etc/sudoers** file: 

```bash
EDITOR=nano visudo
```

Uncommenting the line by removing **\#** character at the beginning of line: 

```bash
# %wheel ALL=(ALL:ALL) ALL
```
