# Creating Synthetic Time Series from Interpretable Components

```python
from collections.abc import Callable
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
```

Firstly, all necessary libraries are included for the notebook. `Callable` from the `collections.abc` module is used for type hinting. Additionally, `numpy` and `matplotlib` libraries are imported for handling numerical operations efficiently, in particular for array creation and manipulation, and visual representation of data, respectively.

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
```

The default settings of `matplotlib` library are customized above by using `rcParams` option so as to enhance the visual presentability of plots. Specifically, it adjusts linewidth, text font, major and minor tick sizes, makes spines visible. The math font is set to `"cm"` (Computer Modern) for **LaTeX**-style equations.

```python
def trend(time_steps: np.ndarray,
          m: float = 1.0,
          n: float = 0.0) -> np.ndarray:
    return m * time_steps + n


def level(time_steps: np.ndarray, n: float = 0.0) -> np.ndarray:
    return trend(time_steps, 0, n)


def periodicity(time_steps: np.ndarray,
                func: Callable[[np.ndarray], np.ndarray],
                phase_shift: float = 0.0,
                period: float = 1.0) -> np.ndarray:
    time_steps = (time_steps + phase_shift) % period
    return func(time_steps)


def noise(func: Callable[[tuple], np.ndarray],
          noise_level: float = 1.0,
          n_points: int = 10,
          seed_n: int = 666) -> np.ndarray:
    np.random.seed(seed_n)
    return noise_level * func(size=(n_points,))


def pattern(time_steps: np.ndarray,
            pattern_func: Callable[[np.ndarray], np.ndarray],
            trend_line: float,
            period: float,
            amplitude: float,
            x_shift: float,
            y_shift: float,
            noise_func: Callable[[tuple], np.ndarray] = np.random.normal,
            noise_level: float = 1.0,
            seed_n: int = 666):
    return trend(time_steps, trend_line, 0) + \
           amplitude * periodicity(time_steps, pattern_func, x_shift, period) + \
           noise(noise_func, noise_level, time_steps.shape[0], seed_n) + \
           level(time_steps, y_shift)


def weird_func_one(x: np.ndarray) -> np.ndarray:
    data = np.piecewise(x,
                        [x / 365 < 0.5, 0.5 <= x / 365],
                        [lambda y: np.cos(2 * np.pi * (y / 365)), lambda y: 1 / np.exp(3 * (y / 365))])
    return data
```

These are main utility functions defined for generating various components of a time series:

* `trend(time_steps, m, n)`: Creates a linear trend by applying line equation formula (`y = m * x + n`), where `m` is the slope and `n` is the y-axis intercept.

* `level(time_steps, n)`: Produces a constant level across the time series by treating it as a zero-slope trend, shifting it up or down by `n` amount.

* `periodicity(time_steps, func, phase_shift, period)`: Adds a repeating pattern to the series using a provided function `func` (e.g., sine or a custom shape). The `phase_shift` adjusts the function"s start point, and `period` defines the cycle length.

* `noise(func, noise_level, n_points, seed_n)`: Introduces randomness into the time series. By default, it uses a stochastic noise function (e.g., `np.random.normal`), scaled by `noise_level`, and optionally seeded with argument `seed_n` for reproducibility.

* `pattern(...)`: This is the high-level function that combines all components defined above into complete synthetic time series.

* `weird_func_one(x)`: A custom, non-standard function used as a example for imitating some complex, periodic shape.

```python
x = np.linspace(0, 700, 70)

y = weird_func_one(x)
y1 = periodicity(x, weird_func_one, 0, 365)
y2 = pattern(x, weird_func_one, 0.5, 365, 60, 0, 0, np.random.normal, 20)

fig, axes = plt.subplots(3, 1, sharex=True)

axes[0].scatter(x, y, s=15, c="red", label=r"$y$")
axes[0].set_ylim(y.min() - 0.5, y.max() + 0.5)
axes[0].set_ylabel(r"$y$", fontsize=16, rotation=0, labelpad=16)
axes[0].legend(fontsize=14, loc="best", edgecolor="black",
               fancybox=False, shadow=True, borderaxespad=1)
axes[0].grid()

axes[1].scatter(x, y1, s=15, c="green", label=r"$y_{1}$")
axes[1].set_ylim(y1.min() - 0.5, y1.max() + 0.5)
axes[1].set_ylabel(r"$y_{1}$", fontsize=16, rotation=0, labelpad=16)
axes[1].legend(fontsize=14, loc="best", edgecolor="black",
               fancybox=False, shadow=True, borderaxespad=1)
axes[1].grid()

axes[2].scatter(x, y2, s=15, c="blue", label=r"$y_{2}$")
axes[2].set_xlabel(r"$x$", fontsize=16, labelpad=10)
axes[2].set_ylim(y2.min() - 50, y2.max() + 50)
axes[2].set_ylabel(r"$y_{2}$", fontsize=16, rotation=0, labelpad=20)
axes[2].legend(fontsize=14, loc="best", edgecolor="black",
               fancybox=False, shadow=True, borderaxespad=1)
axes[2].grid()

plt.tight_layout(pad=2)
plt.show()
```


    
![Figure](codes/python/python_6_files/output_7_0.png)
    


# Conclusion

Here it was explored how to generate synthetic time series data consisting of different components such as trend, seasonality (periodicity), and noise by defining respective custom functions. In other words, the synthetic data is constructed as a linear combination of these components. The results are visualized using scatter plots to illustrate the combined effects. Based on the `weird_func_one`, the top plot shows the raw pattern, a portion of which is reused periodically in the subsequent plots. The second plot displays this repeated segment, forming a periodic pattern. Finally, the bottom plot presents the complete synthetic time series as defined by the `pattern` function.
