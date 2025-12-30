# How to Fix "visudo: no editor found" Error

This error (`visudo: no editor found (editor path = /usr/bin/vi)`) occurs when there is no **vi** text editor installed if `sudo visudo` command is run. It happens even if there is a text editor, say **nano**, been installed before.

Install **vi** editor package (with **sudo** if it was set up):

```bash
sudo pacman -S vi
```
