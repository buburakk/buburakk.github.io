# Updating File Content

If there are two files, namely **file.old** and **file.new**, with partially different contents. Running the command below shows side by side comparison of both files in color:

```bash
diff -y --color=auto file.old file.new
```

The option `-y` enables side by side view and `--color=auto` displays changes between the two files in a visually appealing way. The next command creates a file that takes into account those changes:

```bash
diff -u file.old file.new > file.patch
```

The changes are then applied to the old file via patch command:

```bash
patch file.old < file.patch
```
