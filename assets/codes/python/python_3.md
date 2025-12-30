# Cartesian Products in NumPy

```python
from itertools import product
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
```

```python
def cart_prod_one(variables):
  return np.stack(np.meshgrid(*variables, indexing="ij"), axis=-1).reshape(-1, len(variables))


def cart_prod_two(variables):
  return np.array(list(product(*variables)))
```

```python
x = np.array([0, 1])
y = np.array([7, 77, 777])
z = np.array([-1234, -123, -12, -1])

print(f"'1-dimensional' array: {(cart_prod_one([x]) == cart_prod_two([x])).all()}",
      f"'2-dimensional' array: {(cart_prod_one([x, y]) == cart_prod_two([x, y])).all()}",
      f"'3-dimensional' array: {(cart_prod_one([x, y, z]) == cart_prod_two([x, y, z])).all()}",
      sep="\n\n")
```

    '1-dimensional' array: True
    
    '2-dimensional' array: True
    
    '3-dimensional' array: True


```python
x = np.linspace(-1, 1, 10)
t = np.linspace(0, 1, 100)

d2 = cart_prod_one([x, t])

z1 = np.sin(d2[:, 0] * d2[:, 1]).reshape((10, 100))

print(f"'x1' variable shape: {x.shape}",
      f"'t1' variable shape: {t.shape}",
      f"'z1' variable shape: {z1.shape}",
      sep="\n\n")
```

    'x1' variable shape: (10,)
    
    't1' variable shape: (100,)
    
    'z1' variable shape: (10, 100)


```python
fig, ax = plt.subplots()

img = ax.imshow(z1, aspect="auto", cmap="viridis", extent=(0, 1, -1, 1), origin="lower")

ax.set_xlabel(r"$t$", fontsize=18, labelpad=10)
ax.set_ylabel(r"$x$", fontsize=18, labelpad=10, rotation="horizontal")

cbar = plt.colorbar(img)
cbar.ax.set_title(r"$\sin(xt)$", pad=15)

plt.tight_layout()
plt.show()
```


    
![Figure](assets/codes/python/python_3_files/output_6_0.png)
    


```python
x = np.linspace(-1, 1, 10)
t = np.linspace(0, 1, 100)
X2, T2 = np.meshgrid(*[x, t], indexing="ij")
Z2 = np.sin(X2 * T2)

print(f"'X2' variable shape: {X2.shape}",
      f"'T2' variable shape: {T2.shape}",
      f"'Z2' variable shape: {Z2.shape}",
      sep="\n\n")
```

    'X2' variable shape: (10, 100)
    
    'T2' variable shape: (10, 100)
    
    'Z2' variable shape: (10, 100)


```python
fig, ax = plt.subplots()

img = ax.imshow(Z2, aspect="auto", cmap="viridis", extent=(0, 1, -1, 1), origin="lower")

ax.set_xlabel(r"$t$", fontsize=18, labelpad=10)
ax.set_ylabel(r"$x$", fontsize=18, labelpad=10, rotation="horizontal")

cbar = plt.colorbar(img)
cbar.ax.set_title(r"$\sin(xt)$", pad=15)

plt.tight_layout()
plt.show()
```


    
![Figure](assets/codes/python/python_3_files/output_8_0.png)
    


# Conclusion

**NumPy** library lacks a built-in Cartesian product function. However, it was shown above that there is a way to achieve the same with the library's built-in functions as exemplified above.

We first check if two definitions of Cartesian product match on arbitrary data. Python has a built-in module, namely `itertools`, in which Cartesian product is defined by `product` function. We obtain `cart_prod_two` by combining this with **NumPy**. Similarly, we define `cart_prod_one` by solely utilizing **NumPy** operations above.

A function of two variables, $\sin(x t)$, was plotted on a two-dimensional Cartesian space with the help of `cart_prod_one` function defined while in `cart_prod_two` the function `product`, which is Cartesian product, from Python's built-in `itertools` library was used to obtain the same result that was validated both visually and numerically.
