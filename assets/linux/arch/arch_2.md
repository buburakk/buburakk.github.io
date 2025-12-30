# File Format Error When Running

Sometimes a file format becomes inaccessible when it is directly transferred from one operating system to another. In the case of Linux, if a file, with **.txt** extension, created on Windows operating system is moved onto Linux and ran, an error occurs:

```bash
-bash: /home/username/.local/bin/filename: cannot execute: required file not found
```

This happens as a result of different line ending conventions used in text format for operating systems:

* DOS: carriage return (\\r) and line feed (\\n)  

* Mac: carriage return (\\r)  

* Unix: line feed (\\n)

If the file named **filename** is in DOS Format, it can be fixed by running:

```bash
sed -i 's/\r$//' filename
```

Alternatively, there is a package, namely **dos2unix**, already in use that comes with many other functionalities than simply converting a file in DOS format into Unix format. It can be installed with the command below:

```bash
sudo pacman -S dos2unix
```
