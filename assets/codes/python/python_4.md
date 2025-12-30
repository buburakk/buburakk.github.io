# Automatic Differentiation on 2D Domains with JAX

```python
from jax import grad, hessian, vmap
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
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
def cartesian_product(variables):
  return jnp.stack(jnp.meshgrid(*variables, indexing="ij"), axis=-1).reshape(-1, len(variables))


def get_points(domain):
  return tuple(i.squeeze() for i in jnp.split(domain, domain.shape[1], axis=1))


def nabla(func, *args):
  args = tuple(jnp.asarray([arg]) if not isinstance(arg, jnp.ndarray) else arg for arg in args)
  df = grad(func, range(len(args)))
  return jnp.stack(vmap(df)(*args)).T


def nabla2(func, *args):
  args = tuple(jnp.asarray([arg]) if not isinstance(arg, jnp.ndarray) else arg for arg in args)
  d2f = hessian(func, range(len(args)))
  t = tuple(jnp.stack(i) for i in vmap(d2f)(*args))
  return jnp.stack(t, axis=1).T
```

```python
def f(x, t):
  return 2 * (x ** 3) * t + 7 * x * (t ** 2)


# First-order derivatives
df_dx = lambda xv, tv: jnp.stack([6 * (x ** 2) * t + 7 * (t ** 2) for x, t in zip(xv, tv)])
df_dt = lambda xv, tv: jnp.stack([2 * (x ** 3) + 14 * x * t for x, t in zip(xv, tv)])


# Second-order derivatives
d2f_dx2 = lambda xv, tv: jnp.stack([12 * x * t for x, t in zip(xv, tv)])
d2f_dxdt = lambda xv, tv: jnp.stack([6 * (x ** 2) + 14 * t for x, t in zip(xv, tv)])
d2f_dtdx = d2f_dxdt # Due to symmetry property of mixed partial derivatives
d2f_dt2 = lambda xv, tv: jnp.stack([14 * x for x, t in zip(xv, tv)])


def g(x, y, z, t):
  return x * (y ** 2) * (z ** 3) * (t ** 4)


# First-order derivatives
dg_dx = lambda xv, yv, zv, tv: jnp.stack([(y ** 2) * (z ** 3) * (t ** 4) for x, y, z, t in zip(xv, yv, zv, tv)])
dg_dy = lambda xv, yv, zv, tv: jnp.stack([2 * x * y * (z ** 3) * (t ** 4) for x, y, z, t in zip(xv, yv, zv, tv)])
dg_dz = lambda xv, yv, zv, tv: jnp.stack([3 * x * (y ** 2) * (z ** 2) * (t ** 4) for x, y, z, t in zip(xv, yv, zv, tv)])
dg_dt = lambda xv, yv, zv, tv: jnp.stack([4 * x * (y ** 2) * (z ** 3) * (t ** 3) for x, y, z, t in zip(xv, yv, zv, tv)])


# Second-order derivatives
d2g_dx2 = lambda xv, yv, zv, tv: 0
d2g_dy2 = lambda xv, yv, zv, tv: jnp.stack([2 * x * (z ** 3) * (t ** 4) for x, y, z, t in zip(xv, yv, zv, tv)])
d2g_dz2 = lambda xv, yv, zv, tv: jnp.stack([6 * x * (y ** 2) * z * (t ** 4) for x, y, z, t in zip(xv, yv, zv, tv)])
d2g_dt2 = lambda xv, yv, zv, tv: jnp.stack([12 * x * (y ** 2) * (z ** 3) * (t ** 2) for x, y, z, t in zip(xv, yv, zv, tv)])
d2g_dxdy = lambda xv, yv, zv, tv: jnp.stack([2 * y * (z ** 3) * (t ** 4) for x, y, z, t in zip(xv, yv, zv, tv)])
d2g_dydx = d2g_dxdy # Due to symmetry property of mixed partial derivatives
d2g_dxdz = lambda xv, yv, zv, tv: jnp.stack([3 * (y ** 2) * (z ** 2) * (t ** 4) for x, y, z, t in zip(xv, yv, zv, tv)])
d2g_dzdx = d2g_dxdz # Due to symmetry property of mixed partial derivatives
d2g_dxdt = lambda xv, yv, zv, tv: jnp.stack([4 * (y ** 2) * (z ** 3) * (t ** 3) for x, y, z, t in zip(xv, yv, zv, tv)])
d2g_dtdx = d2g_dxdt # Due to symmetry property of mixed partial derivatives
d2g_dydz = lambda xv, yv, zv, tv: jnp.stack([6 * x * y * (z ** 2) * (t ** 4) for x, y, z, t in zip(xv, yv, zv, tv)])
d2g_dzdy = d2g_dydz # Due to symmetry property of mixed partial derivatives
d2g_dydt = lambda xv, yv, zv, tv: jnp.stack([8 * x * y * (z ** 3) * (t ** 3) for x, y, z, t in zip(xv, yv, zv, tv)])
d2g_dtdy = d2g_dydt # Due to symmetry property of mixed partial derivatives
d2g_dzdt = lambda xv, yv, zv, tv: jnp.stack([12 * x * (y ** 2) * (z ** 2) * (t ** 3) for x, y, z, t in zip(xv, yv, zv, tv)])
d2g_dtdz = d2g_dzdt # Due to symmetry property of mixed partial derivatives
```

```python
num_x, num_t = 123, 123

x = jnp.linspace(-1, 1, num_x)
t = jnp.linspace(-1, 1, num_t)
d = cartesian_product([x, t])
xv, tv = get_points(d)

print(f"'x-dimension' points in domain: {xv.shape}",
      f"'t-dimension' points in domain: {tv.shape}",
      sep="\n\n")
```

    'x-dimension' points in domain: (15129,)
    
    't-dimension' points in domain: (15129,)


```python
fig, ax = plt.subplots()

img = ax.imshow(f(xv, tv).reshape(num_x, num_t).T,
                aspect="auto",
                cmap="viridis",
                extent=(-1, 1, -1, 1),
                origin="lower")

ax.set_xlabel(r"$x$", fontsize=18, labelpad=10)
ax.set_ylabel(r"$t$", fontsize=18, labelpad=10, rotation="horizontal")
ax.set_title(r"$f(x, t) = 2 x^{3} t + 7 x t^{2}$", pad=15)

cbar = plt.colorbar(img)
cbar.ax.set_title(r"$f(x, t)$", pad=15)

plt.tight_layout()
plt.show()
```


    
![Figure](assets/codes/python/python_4_files/output_6_0.png)
    


```python
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

img_0 = ax[0].imshow(nabla(f, xv, tv)[:, 0].reshape(num_x, num_t).T,
                     aspect="auto",
                     cmap="viridis",
                     extent=(-1, 1, -1, 1),
                     origin="lower")

ax[0].set_xlabel(r"$x$", fontsize=18, labelpad=10)
ax[0].set_ylabel(r"$t$", fontsize=18, labelpad=10, rotation="horizontal")
ax[0].set_title(r"$f_{x}(x, t) = 6 x^{2} t + 7 t^{2}$", pad=15)

cbar_0 = plt.colorbar(img_0)
cbar_0.ax.set_title(r"$f_{x}(x, t)$", pad=15)

img_1 = ax[1].imshow(nabla(f, xv, tv)[:, 1].reshape(num_x, num_t).T,
                     aspect="auto",
                     cmap="viridis",
                     extent=(-1, 1, -1, 1),
                     origin="lower")

ax[1].set_xlabel(r"$x$", fontsize=18, labelpad=10)
ax[1].set_ylabel(r"$t$", fontsize=18, labelpad=10, rotation="horizontal")
ax[1].set_title(r"$f_{t}(x, t) = 2 x^{3} + 14 x t$", pad=15)

cbar_1 = plt.colorbar(img_1)
cbar_1.ax.set_title(r"$f_{t}(x, t)$", pad=15)

plt.tight_layout()
plt.show()
```


    
![Figure](assets/codes/python/python_4_files/output_7_0.png)
    


```python
num_x, num_y, num_z, num_t = 2, 2, 2, 2

x = jnp.linspace(-1, 1, num_x)
y = jnp.linspace(-1, 1, num_y)
z = jnp.linspace(-1, 1, num_z)
t = jnp.linspace(-1, 1, num_t)

xvf, tvf = get_points(cartesian_product([x, t]))

df = nabla(f, xvf, tvf)
d2f = nabla2(f, xvf, tvf)

xvg, yvg, zvg, tvg = get_points(cartesian_product([x, y, z, t]))

dg = nabla(g, xvg, yvg, zvg, tvg)
d2g = nabla2(g, xvg, yvg, zvg, tvg)

print(f"'df_dx': {(df_dx(xvf, tvf) == df[:, 0]).all()}",
      f"'df_dt': {(df_dt(xvf, tvf) == df[:, 1]).all()}",
      f"'d2f_dx2': {(d2f_dx2(xvf, tvf) == d2f[:, 0, 0]).all()}",
      f"'d2f_dxdt': {(d2f_dxdt(xvf, tvf) == d2f[:, 0, 1]).all()}",
      f"'d2f_dtdx': {(d2f_dtdx(xvf, tvf) == d2f[:, 1, 0]).all()}",
      f"'d2f_dt2': {(d2f_dt2(xvf, tvf) == d2f[:, 1, 1]).all()}",
      f"'dg_dx': {(dg_dx(xvg, yvg, zvg, tvg) == dg[:, 0]).all()}",
      f"'dg_dy': {(dg_dy(xvg, yvg, zvg, tvg) == dg[:, 1]).all()}",
      f"'dg_dz': {(dg_dz(xvg, yvg, zvg, tvg) == dg[:, 2]).all()}",
      f"'dg_dt': {(dg_dt(xvg, yvg, zvg, tvg) == dg[:, 3]).all()}",
      f"'d2g_dx2': {(d2g_dx2(xvg, yvg, zvg, tvg) == d2g[:, 0, 0]).all()}",
      f"'d2g_dxdy': {(d2g_dxdy(xvg, yvg, zvg, tvg) == d2g[:, 0, 1]).all()}",
      f"'d2g_dxdz': {(d2g_dxdz(xvg, yvg, zvg, tvg) == d2g[:, 0, 2]).all()}",
      f"'d2g_dxdt': {(d2g_dxdt(xvg, yvg, zvg, tvg) == d2g[:, 0, 3]).all()}",
      f"'d2g_dydx': {(d2g_dydx(xvg, yvg, zvg, tvg) == d2g[:, 1, 0]).all()}",
      f"'d2g_dy2': {(d2g_dy2(xvg, yvg, zvg, tvg) == d2g[:, 1, 1]).all()}",
      f"'d2g_dydz': {(d2g_dydz(xvg, yvg, zvg, tvg) == d2g[:, 1, 2]).all()}",
      f"'d2g_dydt': {(d2g_dydt(xvg, yvg, zvg, tvg) == d2g[:, 1, 3]).all()}",
      f"'d2g_dzdx': {(d2g_dzdx(xvg, yvg, zvg, tvg) == d2g[:, 2, 0]).all()}",
      f"'d2g_dzdy': {(d2g_dzdy(xvg, yvg, zvg, tvg) == d2g[:, 2, 1]).all()}",
      f"'d2g_dz2': {(d2g_dz2(xvg, yvg, zvg, tvg) == d2g[:, 2, 2]).all()}",
      f"'d2g_dzdt': {(d2g_dzdt(xvg, yvg, zvg, tvg) == d2g[:, 2, 3]).all()}",
      f"'d2g_dtdx': {(d2g_dtdx(xvg, yvg, zvg, tvg) == d2g[:, 3, 0]).all()}",
      f"'d2g_dtdy': {(d2g_dtdy(xvg, yvg, zvg, tvg) == d2g[:, 3, 1]).all()}",
      f"'d2g_dtdz': {(d2g_dtdz(xvg, yvg, zvg, tvg) == d2g[:, 3, 2]).all()}",
      f"'d2g_dt2': {(d2g_dt2(xvg, yvg, zvg, tvg) == d2g[:, 3, 3]).all()}",
      sep="\n\n")
```

    'df_dx': True
    
    'df_dt': True
    
    'd2f_dx2': True
    
    'd2f_dxdt': True
    
    'd2f_dtdx': True
    
    'd2f_dt2': True
    
    'dg_dx': True
    
    'dg_dy': True
    
    'dg_dz': True
    
    'dg_dt': True
    
    'd2g_dx2': True
    
    'd2g_dxdy': True
    
    'd2g_dxdz': True
    
    'd2g_dxdt': True
    
    'd2g_dydx': True
    
    'd2g_dy2': True
    
    'd2g_dydz': True
    
    'd2g_dydt': True
    
    'd2g_dzdx': True
    
    'd2g_dzdy': True
    
    'd2g_dz2': True
    
    'd2g_dzdt': True
    
    'd2g_dtdx': True
    
    'd2g_dtdy': True
    
    'd2g_dtdz': True
    
    'd2g_dt2': True


# Conclusion

A two-dimensional domain was instantiated with the help of `cartesian_product` and `get_points` helper functions. For each dimension a number of points were sampled before taking Cartesian product of them to define our entire domain of interest before extracting each dimension into their respective variables. `nabla` and `nabla2` functions were used to obtain the first- and second-order partial derivatives of a given function. Their results would be in **JAX** array data type. Alternatively, `vmap(grad(f, (0, 1)))(xv, tv)` for the first-order and `vmap(hessian(f, (0, 1)))(xv, tv)` for the second-order derivatives can be used to obtain the same results in Python tuple in `nabla` and `nabla2` functions, respectively. In our case $123$ points were set each dimension, and this makes up $15129$ points in total that define our domain. Hence, each extracted dimension has $15129$ points.
