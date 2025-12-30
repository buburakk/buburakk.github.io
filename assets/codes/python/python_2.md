# Animation in Python with Matplotlib

```python
from IPython.display import Image, display
from matplotlib.animation import PillowWriter
from matplotlib import animation
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
```

```python
mpl.rcParams["figure.figsize"] = (8, 6)
mpl.rcParams["figure.dpi"] = 90
mpl.rcParams["axes.spines.top"] = True
mpl.rcParams["axes.spines.right"] = True
mpl.rcParams["axes.spines.bottom"] = True
mpl.rcParams["axes.spines.left"] = True
mpl.rcParams["axes.grid"] = True
mpl.rcParams["axes.titlesize"] = 16
mpl.rcParams["axes.linewidth"] = 1.25
mpl.rcParams["axes.labelsize"] = 14
mpl.rcParams["xtick.minor.size"] = 4
mpl.rcParams["xtick.minor.width"] = 1
mpl.rcParams["xtick.major.size"] = 8
mpl.rcParams["xtick.major.width"] = 1.25
mpl.rcParams["xtick.labelsize"] = 14
mpl.rcParams["ytick.minor.size"] = 4
mpl.rcParams["ytick.minor.width"] = 1
mpl.rcParams["ytick.major.size"] = 8
mpl.rcParams["ytick.major.width"] = 1.25
mpl.rcParams["ytick.labelsize"] = 14
mpl.rcParams["legend.fontsize"] = 14
mpl.rcParams["mathtext.fontset"] = "cm"
mpl.rcParams["font.family"] = "Liberation Serif"
mpl.rcParams["animation.html"] = "jshtml" # For animations
```

```python
x = np.linspace(0, 50, 250)

f1 = np.sin((2 * np.pi * x) / 10.0) * np.exp(-x / 10)
f2 = np.cos(x)
```

```python
fig, ax = plt.subplots()

line_plot_one, = ax.plot([], [], "r-", linewidth=1.5, label=r"$f_{1} (x) = y = \sin\left(\frac{2\pi x}{10}\right) e^{-\frac{x}{10}}$")
line_plot_two, = ax.plot([], [], "b-", linewidth=1.5, label=r"$f_{2} (x) = y = \cos(x)$")

ax.set_xlabel(r"$x$", labelpad=10)
ax.set_ylabel(r"$y$", labelpad=10, rotation="horizontal")
ax.legend(loc="best", edgecolor="black",
          fancybox=False, shadow=True, borderaxespad=1)

plt.tight_layout()
plt.close()


def run_animation(i):
  line_plot_one.set_data(x[:i + 1], f1[:i + 1])
  line_plot_two.set_data(x[:i + 1], f2[:i + 1])
  min_f1 = np.min(f1[:i + 1])
  min_f2 = np.min(f2[:i + 1])
  max_f1 = np.max(f1[:i + 1])
  max_f2 = np.max(f2[:i + 1])
  ax.set_xlim(0, 1.25 * np.max(x[:i + 1]) + 1)
  ax.set_ylim(1.25 * min(min_f1, min_f2) - 1, 1.25 * max(max_f1, max_f2) + 1)
  return line_plot_one, line_plot_two


ani = animation.FuncAnimation(fig, run_animation, frames=250, interval=250, blit=True)
writer = PillowWriter(fps=15, bitrate=900)
ani.save("animation.gif", writer=writer)
display(Image("animation.gif"))
```


![Figure](assets/codes/python/python_2_files/output_4_0.gif)


# Conclusion

Creating static plots in **Matplotlib** library is one common practice. Animation on plots is sometimes a better way to convey ideas, key insights. It can be either simulating a physical phenomenon (e.g., projectile motion, or how the price of an asset in stock market). This can be achieved by using `FuncAnimation` class from **Matplotlib** library in Python was demonstrated here.

First, $250$ data points were generated between $0$ and $50$, all of which were stored in $x$ variable as one-dimensional **NumPy** array, then the function values corresponding to those points were similarly stored in variables $f_{1}$ and $f_{2}$. Although it is not necessary, the total number of frames for animation was set to be $250$ in order to match each data point in the arrays. In other words, array indices and frame numbers match exactly. Inside `run_animation` function, the data to be plotted on the figure was chosen by `set_data` option that picks points frame by frame in the arrays from the first index up to index `i` (`i` included). Additionally, the limits (lower and upper limits) of axes were determined by obtaining minimum and maximum values for function values within a specified range (`[:i + 1]`), and were defined by `set_xlim` and `set_ylim` options after adding some offsets to those values.

As the animation progresses, updating axes limits iteratively enables to capture both functions within the frame throughout. It also includes styling the plots with `rcParams`, defining functions, generating data, and finally creating an animation of two functions, $f_{1}(x)$, $f_{2}(x)$, being plotted over variable $x$.
