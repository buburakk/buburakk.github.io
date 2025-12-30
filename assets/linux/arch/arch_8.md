# Updating "man" Database

When `whatis htop` returns:

```bash
htop: nothing appropriate.
```

Update **man** database by running:

```bash
sudo mandb
```

Then `whatis htop` returns:

```bash
htop (1)             - interactive process viewer
```
