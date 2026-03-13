# Finite Difference Differentiation: Accuracy and Error

```python
from itertools import product
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
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
def _factorial(n):
  return np.prod(np.arange(1, n + 1, dtype=np.uint16), dtype=np.uint64)


def _broadcast(arg, num_dims):
  """Broadcast scalar/str arguments to a list with one entry per dimension.
  """
  if isinstance(arg, str) or not hasattr(arg, "__iter__"):
    return [arg] * num_dims
  return list(arg)


def _get_stencil(m, p, method):
  s = m + p
  if method == "centered":
    if s % 2 == 0:
      s += 1
    l = (s - 1) // 2
    i_min, i_max = -l, l
  if method == "forward":
    i_min, i_max = 0, s - 1
  if method == "backward":
    i_min, i_max = -(s - 1), 0
  indices = np.arange(i_min, i_max + 1)
  return indices, s


def _stencil(stencils, is_multivariate, dim, m, p, method):
  """Returns the stencil indices and stencil size for finite difference approximation.
  """
  if stencils is not None:
    if is_multivariate:
      indices = np.asarray(stencils[dim])
      indices, len(indices)
    indices = np.asarray(stencils)
    return indices, len(indices)
  return _get_stencil(m, p, method)


def finite_diff(func, x, n, p=4, h=1e-4, method="centered", stencils=None):
  is_multivariate = isinstance(n, (list, tuple))
  x_vec = np.asarray(x, dtype=np.float32)
  n_vec = (n,) if not is_multivariate else tuple(n)
  num_dims = len(n_vec)
  h_vec = _broadcast(h, num_dims)
  p_vec = _broadcast(p, num_dims)
  method_vec = _broadcast(method, num_dims)
  indices_list, coeffs_list = [], []
  for dim, (m, p_i, method_i) in enumerate(zip(n_vec, p_vec, method_vec)):
    indices, s = _stencil(stencils, is_multivariate, dim, m, p_i, method_i)
    W = np.vander(indices, increasing=True).T
    e = np.zeros(s)
    e[m] = 1.0
    C = np.linalg.solve(W, e)
    indices_list.append(indices)
    coeffs_list.append(C)
  index_grids = np.meshgrid(*indices_list, indexing="ij")
  offsets = np.stack(index_grids, axis=-1) * np.array(h_vec)
  points_to_eval = x_vec + offsets
  f_values = np.apply_along_axis(func, -1, points_to_eval)
  coeff_grids = np.meshgrid(*coeffs_list, indexing="ij")
  K = np.prod(np.array(coeff_grids), axis=0)
  derivative_sum = np.sum(K * f_values)
  m_factorial_prod = np.prod([_factorial(order) for order in n_vec])
  h_power_m_prod = np.prod([h_val ** m_val for h_val, m_val in zip(h_vec, n_vec)])
  if h_power_m_prod == 0: return np.inf
  return (m_factorial_prod / h_power_m_prod) * derivative_sum
```

```python
def f(x):
  return x[0] ** 3


df_dx = lambda x: 3 * (x[0] ** 2)
d2f_dx2 = lambda x: 6 * x[0]
d3f_dx3 = lambda x: 6.0
d4f_dx4 = lambda x: 0.0

results = {"func": [], 'x': [], 'n': [], 'p': [], 'h': [], "error_rel": []}

x_inputs = [[-1.05], [1.0], [1.23], [2.345]]
p_inputs = [1, 2, 3, 4]
h_inputs = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
n_inputs = [(1,), (2,), (3,), (4,)]
approx_functions = [finite_diff] * len(n_inputs)
true_functions = [df_dx, d2f_dx2, d3f_dx3, d4f_dx4]

for x, p, s in product(x_inputs, p_inputs, h_inputs):
  for func_approx, n, func_true in zip(approx_functions, n_inputs, true_functions):
    results["func"].append(f.__name__)
    results['x'].append(x[0])
    results['n'].append(n)
    results['p'].append(p)
    results['h'].append(s)
    true_val = func_true(x)
    approx_val = func_approx(f, x, n, p, s)
    if true_val == 0:
      results["error_rel"].append(np.inf)
    else:
      results["error_rel"].append(np.abs((true_val - approx_val) / true_val) * 100)

results = pd.DataFrame(results)
results["func"] = results["func"].astype("category")
results['n'] = results['n'].astype("category")
unique_x_values = results['x'].unique()
num_x_values = len(unique_x_values)

fig, axes = plt.subplots(1, num_x_values, figsize=(num_x_values * 3, 4), sharey=True)

for i, x_val in enumerate(unique_x_values):
  subset_results = results[results['x'] == x_val]
  sns.lineplot(data=subset_results,
               ax=axes[i],
               palette=sns.color_palette("bright", len(n_inputs)),
               x='h',
               y="error_rel",
               hue='n',
               style='p',
               markers=True,
               markersize=7,
               dashes=False)
  axes[i].set_xscale("log")
  axes[i].set_xticks(h_inputs)
  axes[i].set_xlabel("Step size (h)", labelpad=15)
  axes[i].set_ylabel('')
  axes[i].set_title(f"x = {x_val}", fontsize=14, pad=15)
  axes[i].get_legend().remove()
axes[0].set_yscale("log")
axes[0].set_ylabel(r"$\delta_{rel} \ (\%)$", labelpad=30, rotation="horizontal")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels,
           loc="lower left", mode="expand", edgecolor="black",
           fancybox=False, shadow=True,
           bbox_to_anchor=(axes[0].get_position().x0 - 0.003, axes[0].get_position().y1 + 0.145, 0.8575, 0.102), borderaxespad=0,
           ncol=len(results['p'].unique()) + len(results['n'].unique()) + 2)

plt.tight_layout()
plt.show()
```


    
![Figure](codes/python/python_10_files/output_4_0.png)
    


```python
def g(x):
  return (x[0] ** 2) * (x[1] ** 3)


dg_dx = lambda x: (2 * x[0]) * (x[1] ** 3)
dg_dy = lambda x: (x[0] ** 2) * 3 * (x[1] ** 2)
d2g_dx2 = lambda x: 2 * (x[1] ** 3)
d2g_dxdy = lambda x: (2 * x[0]) * 3 * (x[1] ** 2)
d2g_dy2 = lambda x: (x[0] ** 2) * 6 * x[1]

results = {"func": [], 'x': [], 'n': [], 'p': [], 'h': [], "error_rel": []}

x_inputs = [[-1.05, 0.75], [1.0, 1.0], [1.23, 1.0], [0.0001, 2.345]]
p_inputs = [1, 2, 3, 4]
h_inputs = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
n_inputs = [(1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]
approx_functions = [finite_diff] * len(n_inputs)
true_functions = [dg_dx, dg_dy, d2g_dx2, d2g_dxdy, d2g_dy2]

for x, p, s in product(x_inputs, p_inputs, h_inputs):
  for func_approx, n, func_true in zip(approx_functions, n_inputs, true_functions):
    results["func"].append(g.__name__)
    results['x'].append(x)
    results['n'].append(n)
    results['p'].append(p)
    results['h'].append(s)
    true_val = func_true(x)
    approx_val = func_approx(g, x, n, p, s)
    if true_val == 0:
      results["error_rel"].append(np.inf)
    else:
      results["error_rel"].append(np.abs((true_val - approx_val) / true_val) * 100)

results = pd.DataFrame(results)
results["func"] = results["func"].astype("category")
results['x'] = results['x'].apply(tuple).astype("category")
results['n'] = results['n'].astype("category")
unique_x_values = results['x'].unique()
num_x_values = len(unique_x_values)

fig, axes = plt.subplots(1, num_x_values, figsize=(num_x_values * 3, 4), sharey=True)

for i, x_val in enumerate(unique_x_values):
  subset_results = results[results['x'] == x_val]
  sns.lineplot(data=subset_results,
               ax=axes[i],
               palette=sns.color_palette("bright", len(n_inputs)),
               x='h',
               y="error_rel",
               hue='n',
               style='p',
               markers=True,
               markersize=7,
               dashes=False)
  axes[i].set_xscale("log")
  axes[i].set_xticks(h_inputs)
  axes[i].set_xlabel("Step size (h)", labelpad=15)
  axes[i].set_ylabel('')
  axes[i].set_title(f"x = {x_val}", fontsize=14, pad=15)
  axes[i].get_legend().remove()
axes[0].set_yscale("log")
axes[0].set_ylabel(r"$\delta_{rel} \ (\%)$", labelpad=30, rotation="horizontal")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels,
           loc="lower left", mode="expand", edgecolor="black",
           fancybox=False, shadow=True,
           bbox_to_anchor=(axes[0].get_position().x0 - 0.003, axes[0].get_position().y1 + 0.145, 0.8575, 0.102), borderaxespad=0,
           ncol=len(results['p'].unique()) + len(results['n'].unique()) - 2)

plt.tight_layout()
plt.show()
```


    
![Figure](codes/python/python_10_files/output_5_0.png)
    


```python
def h(x):
  return (2 * ((x[1] - 1) ** 2) * (x[0] ** 3) + 7 * ((x[2] - 2) ** 3) * (x[0] * x[1]) ** 2) * x[3]


dh_dx = lambda x: 2 * x[3] * x[0] * (3 * x[0] * (x[1] - 1) ** 2 + 7 * (x[1] ** 2) * (x[2] - 2) ** 3)
dh_dy = lambda x: 2 * x[3] * (x[0] ** 2) * (2 * x[0] * (x[1] - 1) + 7 * x[1] * (x[2] - 2) ** 3)
dh_dz = lambda x: 21 * x[3] * (x[0] ** 2) * (x[1] ** 2) * (x[2] - 2) ** 2
dh_dt = lambda x: (x[0] ** 2) * (2 * x[0] * (x[1] - 1) ** 2 + 7 * (x[1] ** 2) * (x[2] - 2) ** 3)
d2h_dx2 = lambda x: 2 * x[3] * (6 * x[0] * (x[1] - 1) ** 2 + 7 * (x[1] ** 2) * (x[2] - 2) ** 3)
d2h_dxdy = lambda x: 4 * x[3] * x[0] * (3 * x[0] * (x[1] - 1) + 7 * x[1] * (x[2] - 2) ** 3)
d2h_dxdz = lambda x: 42 * x[3] * x[0] * (x[1] ** 2) * (x[2] - 2) ** 2
d2h_dxdt = lambda x: 2 * x[0] * (3 * x[0] * (x[1] - 1) ** 2 + 7 * (x[1] ** 2) * (x[2] - 2) ** 3)
d2h_dy2 = lambda x: 2 * x[3] * (x[0] ** 2) * (2 * x[0] + 7 * (x[2] - 2) ** 3)
d2h_dydz = lambda x: 42 * x[3] * (x[0] ** 2) * x[1] * (x[2] - 2) ** 2
d2h_dydt = lambda x: (x[0] ** 2) * (4 * x[0] * (x[1] - 1) + 14 * x[1] * (x[2] - 2) ** 3)
d2h_dz2 = lambda x: 42 * x[3] * (x[0] ** 2) * (x[1] ** 2) * (x[2] - 2)
d2h_dzdt = lambda x: 21 * (x[0] ** 2) * (x[1] ** 2) * (x[2] - 2) ** 2
d2h_dt2 = lambda x: 0.0

results = {"func": [], 'x': [], 'n': [], 'p': [], 'h': [], "error_rel": []}

x_inputs = [[-1.05, 0.75, -1.05, 0.75], [1.0, 1.0, 1.0, 1.0],
            [1.23, 1.0, 1.23, 1.0], [0.0001, 2.345, 0.24, 0.1235]]
p_inputs = [1, 2, 3, 4]
h_inputs = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
n_inputs = [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1), (2, 0, 0, 0),
            (1, 1, 0, 0), (1, 0, 1, 0), (1, 0, 0, 1), (0, 2, 0, 0), (0, 1, 1, 0),
            (0, 1, 0, 1), (0, 0, 2, 0), (0, 0, 1, 1), (0, 0, 0, 2)]
approx_functions = [finite_diff] * len(n_inputs)
true_functions = [dh_dx, dh_dy, dh_dz, dh_dt, d2h_dx2, d2h_dxdy, d2h_dxdz,
                  d2h_dxdt, d2h_dy2, d2h_dydz, d2h_dydt, d2h_dz2, d2h_dzdt, d2h_dt2]

for x, p, s in product(x_inputs, p_inputs, h_inputs):
  for func_approx, n, func_true in zip(approx_functions, n_inputs, true_functions):
    results["func"].append(h.__name__)
    results['x'].append(x)
    results['n'].append(n)
    results['p'].append(p)
    results['h'].append(s)
    true_val = func_true(x)
    approx_val = func_approx(h, x, n, p, s)
    if true_val == 0:
      results["error_rel"].append(np.inf)
    else:
      results["error_rel"].append(np.abs((true_val - approx_val) / true_val) * 100)

results = pd.DataFrame(results)
results["func"] = results["func"].astype("category")
results['x'] = results['x'].apply(tuple).astype("category")
results['n'] = results['n'].astype("category")
unique_x_values = results['x'].unique()
num_x_values = len(unique_x_values)

fig, axes = plt.subplots(1, num_x_values, figsize=(num_x_values * 3, 4), sharey=True)

for i, x_val in enumerate(unique_x_values):
  subset_results = results[results['x'] == x_val]
  sns.lineplot(data=subset_results,
               ax=axes[i],
               palette=sns.color_palette("bright", len(n_inputs)),
               x='h',
               y="error_rel",
               hue='n',
               style='p',
               markers=True,
               markersize=7,
               dashes=False)
  axes[i].set_xscale("log")
  axes[i].set_xticks(h_inputs)
  axes[i].set_xlabel("Step size (h)", labelpad=15)
  axes[i].set_ylabel('')
  axes[i].set_title(f"x = {x_val}", fontsize=14, pad=15)
  axes[i].get_legend().remove()
axes[0].set_yscale("log")
axes[0].set_ylabel(r"$\delta_{rel} \ (\%)$", labelpad=30, rotation="horizontal")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels,
           loc="lower left", mode="expand", edgecolor="black",
           fancybox=False, shadow=True,
           bbox_to_anchor=(axes[0].get_position().x0 - 0.0035, axes[0].get_position().y1 + 0.145, 0.8575, 0.102), borderaxespad=0,
           ncol=7)

plt.tight_layout()
plt.show()
```


    
![Figure](codes/python/python_10_files/output_6_0.png)
    


# Conclusion

Derivatives of single- and multi-variable functions at four arbitrarily chosen points were approximated in this study with respect to step size (`h`), order of accuracy (`p`), and type of derivative (`n`) parameters. A higher `p` value implies more stencil points and, hence, in general allowing for higher accuracy calculations. Additionally, `"central"` difference method with uniform grid spacing was chosen by default in the determination of stencil points. The derivative parameter represents order and variable to take derivative with respect to in tuple format. That is, if a function $f(x, y, z)$ has three independent variables, $(1, 0, 0)$ implies the partial derivative with respect to $x$ variable. Similarly, in order to find a mixed derivative, say $\frac{\partial f}{\partial x \partial y \partial z}$, we need $(1, 1, 1)$. `finite_diff` function was defined based on [finite difference](https://www.geometrictools.com/Documentation/FiniteDifferences.pdf) approach to obtain those derivatives. The comparison was made by relative error ($\delta_{rel}$) criterion and plotted in three figures above for three functions against a combination of aforementioned parameters. The following results drawn upon the study can be listed:

* There is no optimal step size and order of accuracy in finite difference calculations for derivatives; and it varies from one function to another.

* There is no guarantee that a higher order of accuracy means a lower relative error.

* The relative error does not respond monotonically to a proportionate change in one of the parameters at all times.
