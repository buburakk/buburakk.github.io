# Efficient Multivariate Taylor Expansions with JAX

Let $f : \mathbb{R}^{n} \to \mathbb{R}$ be a $k$-times continuously differentiable function at the point $\mathbf{a} \in \mathbb{R}^{n}$, then the function can be approximated by [Taylor series](https://en.wikipedia.org/wiki/Taylor%27s_theorem#Taylor's_theorem_for_multivariate_functions) given below given in [multi-index notation](https://en.wikipedia.org/wiki/Multi-index_notation)

$$
f(\mathbf{x}) = \sum_{|\alpha | \leq k}{\frac{D^{\alpha} f(\mathbf{a})}{\alpha !}} (\mathbf{x} - \mathbf{a})^{\alpha},
$$

where

$$
\begin{aligned}
D^{\alpha} f(\mathbf{a}) &= {\frac{\partial^{|\alpha|}f(\mathbf{a})}{\partial \mathbf{x}^{\alpha}}} = {\frac{\partial^{\alpha_{1} + \cdots + \alpha_{n}} f(\mathbf{a})}{\partial x_{1}^{\alpha_{1}} \cdots \partial x_{n}^{\alpha_{n}}}}, \newline
\newline
|\alpha| &= \alpha_{1} + \cdots + \alpha_{n}, \newline
\newline
\alpha ! &= \alpha_{1} ! \cdots \alpha_{n} !, \newline
\newline
(\mathbf{x} - \mathbf{a})^{\alpha} &= (x_{1} - a_{1})^{\alpha_{1}} \cdots (x_{n} - a_{n})^{\alpha_{n}}
\end{aligned}
$$

for $\alpha \in \mathbb{N}^{n}$ and $\mathbf{x} \in \mathbb{R}^{n}$. Unlike simple indexing, this time we have indices that satisfy the inequality ($\alpha_{1} + \alpha_{2} + \cdots + \alpha_{n} \leq k$). For example, if we want to obtain the third-order approximation of a function of two-variable $f(x_{1}, x_{2})$. Given $(\alpha_{1}, \alpha_{2})$ and $k = 3$, we have to solve

$$
\alpha_{1} + \alpha_{2} \leq 3
$$

for unique $(\alpha_{1}, \alpha_{2})$ pairs. Hence, we obtain them as $(0, 0)$, $(0, 1)$, $(0, 2)$, $(0, 3)$, $(1, 0)$, $(1, 1)$, $(1, 2)$, $(2, 0)$, $(2, 1)$, $(3, 0)$. After we plug them into their respective formulas in the series expansion, we have

$$
\begin{aligned}
f(\mathbf{x}) &\approx \frac{f(\mathbf{a})}{0! 0!} (x_{1} - a_{1})^{0} (x_{2} - a_{2})^{0} \newline
\newline
&\quad + \frac{f_{x}(\mathbf{a})}{1! 0!} (x_{1} - a_{1})^{1} (x_{2} - a_{2})^{0} + \frac{f_{y}(\mathbf{a})}{0! 1!} (x_{1} - a_{1})^{0} (x_{2} - a_{2})^{1} \newline
\newline
&\quad + \frac{f_{xx}(\mathbf{a})}{2! 0!} (x_{1} - a_{1})^{2} (x_{2} - a_{2})^{0} + \frac{f_{xy}(\mathbf{a})}{1! 1!} (x_{1} - a_{1})^{1} (x_{2} - a_{2})^{1} + \frac{f_{yy}(\mathbf{a})}{0! 2!} (x_{1} - a_{1})^{0} (x_{2} - a_{2})^{2} \newline
\newline
&\quad + \frac{f_{xxx}(\mathbf{a})}{3! 0!} (x_{1} - a_{1})^{3} (x_{2} - a_{2})^{0} + \frac{f_{xxy}(\mathbf{a})}{2! 1!} (x_{1} - a_{1})^{2} (x_{2} - a_{2})^{1} \newline
\newline
&\quad + \frac{f_{xyy}(\mathbf{a})}{1! 2!} (x_{1} - a_{1})^{1} (x_{2} - a_{2})^{2} + \frac{f_{yyy}(\mathbf{a})}{0! 3!} (x_{1} - a_{1})^{0} (x_{2} - a_{2})^{3}.
\end{aligned}
$$

[Alternatively](https://en.wikipedia.org/wiki/Taylor_series#Taylor_series_in_several_variables), we can expand the series as

$$
\begin{aligned}
f(\mathbf{x}) &\approx f(\mathbf{a}) \newline
\newline
&\quad + f_{x}(\mathbf{a}) (x_{1} - a_{1}) + f_{y}(\mathbf{a}) (x_{2} - a_{2}) \newline
\newline
&\quad + \frac{1}{2!} \left[f_{xx}(\mathbf{a}) (x_{1} - a_{1})^{2} + 2 f_{xy}(\mathbf{a}) (x_{1} - a_{1}) (x_{2} - a_{2}) + f_{yy}(\mathbf{a}) (x_{2} - a_{2})^{2}\right] \newline
\newline
&\quad + \frac{1}{3!} \biggr[f_{xxx}(\mathbf{a}) (x_{1} - a_{1})^{3} + 3 f_{xxy}(\mathbf{a}) (x_{1} - a_{1})^{2} (x_{2} - a_{2})\bigr. \newline
\newline
&\qquad \qquad \left. + 3 f_{xyy}(\mathbf{a}) (x_{1} - a_{1}) (x_{2} - a_{2})^{2} + f_{yyy}(\mathbf{a}) (x_{2} - a_{2})^{3}\right].
\end{aligned}
$$

The advantage of both series expansions lies in the calculation of higher-order derivatives. Since it was stated "$k$-times continuously differentiable" as an assumption at the beginning, for a given order of derivative there is no need to calculate some higher-order derivatives. That is, if $f_{xy}(\mathbf{a})$ is known, calculating $f_{yx}(\mathbf{a})$ becomes redundant. Therefore, calculating only diagonal and upper diagonal of Hessian matrix suffices for the second-order derivatives. This holds for other mixed derivatives too.

```python
from itertools import accumulate, chain, product, combinations_with_replacement
from math import prod, comb, factorial
from collections import Counter
from jax import config, jvp, vjp, grad, vmap, jacrev
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
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
def fi(x):
  return (2 * ((x[1] - 1) ** 2) * (x[0] ** 3) + 7 * ((x[2] - 2) ** 3) * (x[0] * x[1]) ** 2) * x[3]


def fis(x):
  return jnp.sum((2 * ((x[1] - 1) ** 2) * (x[0] ** 3) + 7 * ((x[2] - 2) ** 3) * (x[0] * x[1]) ** 2) * x[3])


def fv(x, y, z, t):
  return (2 * ((y - 1) ** 2) * (x ** 3) + 7 * ((z - 2) ** 3) * (x * y) ** 2) * t


def fvs(x, y, z, t):
  return jnp.sum((2 * ((y - 1) ** 2) * (x ** 3) + 7 * ((z - 2) ** 3) * (x * y) ** 2) * t)
```

```python
def coefficients(counts):
  return prod(map(comb, accumulate(counts), counts))


def multinomials(v, items):
  return prod(v[i] ** j for i, j in items)


def _flatten(iterable):
  if isinstance(iterable[0], (list, tuple)):
    return list(chain.from_iterable(iterable))
  return list(iterable)


def outer(x, y):
  return [[xi * yj for yj in y] for xi in x]


def multinomials_v1(a, n):
  if n == 0:
    return [a[0] ** 0]
  out = list(a)
  for _ in range(1, n):
    out = outer(_flatten(out), a)
  return _flatten(out)


def multinomials_v2(a, n):
  if n == 0:
    return [a[0] ** 0]
  return [prod(combo) for combo in product(a, repeat=n)]


def derivatives_v1(f, indices):
  df = f
  for i in indices:
    df = grad(df, i)
  return df


def derivatives(f, x, indices):
  vm = vmap(derivatives_v1(f, indices))
  for _ in range(len(x) - 1):
    vm = vmap(vm)
  return vm(*x)


def derivatives_v2(f, x, order):
  df = f
  for _ in range(order):
    df = jacrev(df)
  return df(x)
```

```python
def multi_indices(order, dims):
  for o in range(order + 1):
    for i in combinations_with_replacement(range(dims), o):
      yield i


def multi_indices_v1(order, dims):
  for o in range(order + 1):
    for i in product(range(dims), repeat=o):
      yield i, Counter(i)


def multi_indices_v2(order, dims):
  for o in range(order + 1):
    for i in combinations_with_replacement(range(dims), o):
      yield i, Counter(i)
```

```python
def factorials_jax(n):
  return jnp.prod(jnp.arange(1, n + 1))


def coefficients_jax(counts):
  if len(counts) == 0:
    return jnp.array(1)
  return factorials_jax(counts.sum()) / jnp.array([factorials_jax(c) for c in counts]).prod()


def multinomials_jax(v, indices, counts):
  if len(indices) == 0:
    return jnp.array(1.0)
  return jnp.prod(jnp.power(v[indices], counts))


def multinomials_jax_v1(a, n):
  if n == 0:
    return jnp.array([1.0])
  out = a
  for _ in range(1, n):
    out = jnp.outer(out, a)
  return out


def multinomials_jax_v2(a, n):
  if n == 0:
    return jnp.array([1.0])
  eins = a
  for i in range(1, n):
    eins = jnp.einsum(eins,
                      list(range(i)),
                      a,
                      [i],
                      list(range(i + 1)),
                      optimize="greedy")
  return eins


def multinomials_jax_v3(a, n):
  if n == 0:
    return jnp.array([1.0])
  if n == 1:
    return jnp.asarray(a)
  return jnp.einsum(multinomials_jax_v3(a, n - 1),
                    list(range(n - 1)),
                    a,
                    [n - 1],
                    list(range(n)),
                    optimize="greedy")


def multinomials_jax_v4(a, n):
  if n == 0:
    return jnp.array([1.0])
  if n == 1:
    return jnp.asarray(a)
  c = a
  for _ in range(n - 1):
    result = []
    r = [i * j for i in a for j in c]
    result.extend(r)
    c = r
  return jnp.array(result)
```

```python
def taylor_components(f, x, a, order):
  v = tuple(xi - ai for xi, ai in zip(x, a))
  for indices, count in multi_indices_v1(order, len(a)):
    yield indices, \
          1 / factorial(len(indices)), \
          coefficients(count.values()), \
          derivatives_v1(f, indices)(*a), \
          multinomials(v, count.items())


def taylor_jax(f, x, a, order):
  v = jnp.asarray(x) - jnp.asarray(a)
  approx = 0
  for indices in multi_indices(order, len(a)):
    i, j = jnp.unique(jnp.asarray(indices), return_counts=True)
    approx += 1 / factorials_jax(len(indices)) * coefficients_jax(j) * derivatives_v1(f, indices)(*a) * multinomials_jax(v, i, j)
  return approx


def taylor_v1(f, x, a, order):
  v = jnp.asarray(x) - jnp.asarray(a)
  return jnp.sum(jnp.asarray([1 / factorial(len(indices)) * \
                              coefficients(count.values()) * \
                              derivatives_v1(f, indices)(*a) * \
                              jnp.prod(jnp.asarray([v[i] ** j for i, j in count.items()]))
                              for indices, count in multi_indices_v2(order, len(a))]))


def taylor_v2(f, x, a, order):
  v = tuple(xi - ai for xi, ai in zip(x, a))
  return jnp.sum(jnp.asarray([1 / factorial(len(indices)) * \
                              coefficients(count.values()) * \
                              derivatives(f, a, indices) * \
                              jnp.prod(jnp.asarray([v[i] ** j for i, j in count.items()]))
                              for indices, count in multi_indices_v2(order, len(a))]))


def taylor_v3(f, x, a, order):
  return jnp.sum(jnp.asarray([(1 / factorial(o)) * \
                              jnp.dot(derivatives_v2(f, a, o).ravel(), multinomials_jax_v1(x - a, o).ravel())
                              for o in range(order + 1)]))
```

```python
a = jnp.array([0.2])
b = jnp.array([0.3])
c = jnp.array([0.5])
d = jnp.array([0.7])
av, bv, cv, dv = jnp.meshgrid(*[a, b, c, d], indexing="ij")
xv = jnp.zeros_like(av)
```

```python
for e in taylor_components(fv, [0.0, 0.0, 0.0, 0.0], [0.2, 0.3, 0.5, 0.7], 3):
  print(f"(x: {0}, y: {1}, z: {2}, t: {3})",
        f"- Variable Indices: {e[0]}",
        f"- 1 / n!: {e[1]}",
        f"- Coefficients: {e[2]}",
        f"- Derivatives ({len(e[0])}-order): {e[3]}",
        f"- (x - a) ^ α: {e[4]}",
        end="\n\n")
```

    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: () - 1 / n!: 1.0 - Coefficients: 1 - Derivatives (0-order): -0.054047 - (x - a) ^ α: 1
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (0,) - 1 / n!: 1.0 - Coefficients: 1 - Derivatives (1-order): -0.5130300521850586 - (x - a) ^ α: -0.2
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (1,) - 1 / n!: 1.0 - Coefficients: 1 - Derivatives (1-order): -0.4125800132751465 - (x - a) ^ α: -0.3
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (2,) - 1 / n!: 1.0 - Coefficients: 1 - Derivatives (1-order): 0.11906998604536057 - (x - a) ^ α: -0.5
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (3,) - 1 / n!: 1.0 - Coefficients: 1 - Derivatives (1-order): -0.07721000164747238 - (x - a) ^ α: -0.7
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (0, 0) - 1 / n!: 0.5 - Coefficients: 1 - Derivatives (2-order): -2.153550386428833 - (x - a) ^ α: 0.04000000000000001
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (0, 1) - 1 / n!: 0.5 - Coefficients: 2 - Derivatives (2-order): -4.204200267791748 - (x - a) ^ α: 0.06
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (0, 2) - 1 / n!: 0.5 - Coefficients: 2 - Derivatives (2-order): 1.1907000541687012 - (x - a) ^ α: 0.1
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (0, 3) - 1 / n!: 0.5 - Coefficients: 2 - Derivatives (2-order): -0.7329000234603882 - (x - a) ^ α: 0.13999999999999999
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (1, 0) - 1 / n!: 0.5 - Coefficients: 2 - Derivatives (2-order): -4.204200267791748 - (x - a) ^ α: 0.06
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (1, 1) - 1 / n!: 0.5 - Coefficients: 1 - Derivatives (2-order): -1.3006000518798828 - (x - a) ^ α: 0.09
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (1, 2) - 1 / n!: 0.5 - Coefficients: 2 - Derivatives (2-order): 0.7938000559806824 - (x - a) ^ α: 0.15
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (1, 3) - 1 / n!: 0.5 - Coefficients: 2 - Derivatives (2-order): -0.589400053024292 - (x - a) ^ α: 0.21
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (2, 0) - 1 / n!: 0.5 - Coefficients: 2 - Derivatives (2-order): 1.1907001733779907 - (x - a) ^ α: 0.1
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (2, 1) - 1 / n!: 0.5 - Coefficients: 2 - Derivatives (2-order): 0.7938000559806824 - (x - a) ^ α: 0.15
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (2, 2) - 1 / n!: 0.5 - Coefficients: 1 - Derivatives (2-order): -0.1587599813938141 - (x - a) ^ α: 0.25
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (2, 3) - 1 / n!: 0.5 - Coefficients: 2 - Derivatives (2-order): 0.17010000348091125 - (x - a) ^ α: 0.35
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (3, 0) - 1 / n!: 0.5 - Coefficients: 2 - Derivatives (2-order): -0.7329000234603882 - (x - a) ^ α: 0.13999999999999999
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (3, 1) - 1 / n!: 0.5 - Coefficients: 2 - Derivatives (2-order): -0.589400053024292 - (x - a) ^ α: 0.21
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (3, 2) - 1 / n!: 0.5 - Coefficients: 2 - Derivatives (2-order): 0.17010000348091125 - (x - a) ^ α: 0.35
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (3, 3) - 1 / n!: 0.5 - Coefficients: 1 - Derivatives (2-order): 0.0 - (x - a) ^ α: 0.48999999999999994
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (0, 0, 0) - 1 / n!: 0.16666666666666666 - Coefficients: 1 - Derivatives (3-order): 4.116000175476074 - (x - a) ^ α: -0.008000000000000002
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (0, 0, 1) - 1 / n!: 0.16666666666666666 - Coefficients: 3 - Derivatives (3-order): -22.19700050354004 - (x - a) ^ α: -0.012000000000000002
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (0, 0, 2) - 1 / n!: 0.16666666666666666 - Coefficients: 3 - Derivatives (3-order): 5.953500270843506 - (x - a) ^ α: -0.020000000000000004
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (0, 0, 3) - 1 / n!: 0.16666666666666666 - Coefficients: 3 - Derivatives (3-order): -3.0764999389648438 - (x - a) ^ α: -0.028000000000000004
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (0, 1, 0) - 1 / n!: 0.16666666666666666 - Coefficients: 3 - Derivatives (3-order): -22.19700050354004 - (x - a) ^ α: -0.012000000000000002
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (0, 1, 1) - 1 / n!: 0.16666666666666666 - Coefficients: 3 - Derivatives (3-order): -12.894000053405762 - (x - a) ^ α: -0.018
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (0, 1, 2) - 1 / n!: 0.16666666666666666 - Coefficients: 6 - Derivatives (3-order): 7.938000202178955 - (x - a) ^ α: -0.03
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (0, 1, 3) - 1 / n!: 0.16666666666666666 - Coefficients: 6 - Derivatives (3-order): -6.00600004196167 - (x - a) ^ α: -0.041999999999999996
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (0, 2, 0) - 1 / n!: 0.16666666666666666 - Coefficients: 3 - Derivatives (3-order): 5.953500747680664 - (x - a) ^ α: -0.020000000000000004
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (0, 2, 1) - 1 / n!: 0.16666666666666666 - Coefficients: 6 - Derivatives (3-order): 7.938000679016113 - (x - a) ^ α: -0.03
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (0, 2, 2) - 1 / n!: 0.16666666666666666 - Coefficients: 3 - Derivatives (3-order): -1.5875999927520752 - (x - a) ^ α: -0.05
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (0, 2, 3) - 1 / n!: 0.16666666666666666 - Coefficients: 6 - Derivatives (3-order): 1.7010000944137573 - (x - a) ^ α: -0.06999999999999999
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (0, 3, 0) - 1 / n!: 0.16666666666666666 - Coefficients: 3 - Derivatives (3-order): -3.0764999389648438 - (x - a) ^ α: -0.028000000000000004
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (0, 3, 1) - 1 / n!: 0.16666666666666666 - Coefficients: 6 - Derivatives (3-order): -6.00600004196167 - (x - a) ^ α: -0.041999999999999996
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (0, 3, 2) - 1 / n!: 0.16666666666666666 - Coefficients: 6 - Derivatives (3-order): 1.7009999752044678 - (x - a) ^ α: -0.06999999999999999
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (0, 3, 3) - 1 / n!: 0.16666666666666666 - Coefficients: 3 - Derivatives (3-order): 0.0 - (x - a) ^ α: -0.09799999999999999
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (1, 0, 0) - 1 / n!: 0.16666666666666666 - Coefficients: 3 - Derivatives (3-order): -22.19700050354004 - (x - a) ^ α: -0.012000000000000002
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (1, 0, 1) - 1 / n!: 0.16666666666666666 - Coefficients: 3 - Derivatives (3-order): -12.894000053405762 - (x - a) ^ α: -0.018
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (1, 0, 2) - 1 / n!: 0.16666666666666666 - Coefficients: 6 - Derivatives (3-order): 7.938000202178955 - (x - a) ^ α: -0.03
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (1, 0, 3) - 1 / n!: 0.16666666666666666 - Coefficients: 6 - Derivatives (3-order): -6.00600004196167 - (x - a) ^ α: -0.041999999999999996
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (1, 1, 0) - 1 / n!: 0.16666666666666666 - Coefficients: 3 - Derivatives (3-order): -12.894000053405762 - (x - a) ^ α: -0.018
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (1, 1, 1) - 1 / n!: 0.16666666666666666 - Coefficients: 1 - Derivatives (3-order): 0.0 - (x - a) ^ α: -0.026999999999999996
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (1, 1, 2) - 1 / n!: 0.16666666666666666 - Coefficients: 3 - Derivatives (3-order): 2.6460001468658447 - (x - a) ^ α: -0.045
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (1, 1, 3) - 1 / n!: 0.16666666666666666 - Coefficients: 3 - Derivatives (3-order): -1.8580001592636108 - (x - a) ^ α: -0.063
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (1, 2, 0) - 1 / n!: 0.16666666666666666 - Coefficients: 6 - Derivatives (3-order): 7.938000679016113 - (x - a) ^ α: -0.03
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (1, 2, 1) - 1 / n!: 0.16666666666666666 - Coefficients: 3 - Derivatives (3-order): 2.6460001468658447 - (x - a) ^ α: -0.045
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (1, 2, 2) - 1 / n!: 0.16666666666666666 - Coefficients: 3 - Derivatives (3-order): -1.05840003490448 - (x - a) ^ α: -0.075
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (1, 2, 3) - 1 / n!: 0.16666666666666666 - Coefficients: 6 - Derivatives (3-order): 1.1340000629425049 - (x - a) ^ α: -0.105
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (1, 3, 0) - 1 / n!: 0.16666666666666666 - Coefficients: 6 - Derivatives (3-order): -6.00600004196167 - (x - a) ^ α: -0.042
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (1, 3, 1) - 1 / n!: 0.16666666666666666 - Coefficients: 3 - Derivatives (3-order): -1.8580000400543213 - (x - a) ^ α: -0.063
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (1, 3, 2) - 1 / n!: 0.16666666666666666 - Coefficients: 6 - Derivatives (3-order): 1.1340000629425049 - (x - a) ^ α: -0.105
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (1, 3, 3) - 1 / n!: 0.16666666666666666 - Coefficients: 3 - Derivatives (3-order): 0.0 - (x - a) ^ α: -0.14699999999999996
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (2, 0, 0) - 1 / n!: 0.16666666666666666 - Coefficients: 3 - Derivatives (3-order): 5.953500747680664 - (x - a) ^ α: -0.020000000000000004
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (2, 0, 1) - 1 / n!: 0.16666666666666666 - Coefficients: 6 - Derivatives (3-order): 7.938000679016113 - (x - a) ^ α: -0.03
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (2, 0, 2) - 1 / n!: 0.16666666666666666 - Coefficients: 3 - Derivatives (3-order): -1.5875999927520752 - (x - a) ^ α: -0.05
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (2, 0, 3) - 1 / n!: 0.16666666666666666 - Coefficients: 6 - Derivatives (3-order): 1.7010000944137573 - (x - a) ^ α: -0.06999999999999999
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (2, 1, 0) - 1 / n!: 0.16666666666666666 - Coefficients: 6 - Derivatives (3-order): 7.938000679016113 - (x - a) ^ α: -0.03
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (2, 1, 1) - 1 / n!: 0.16666666666666666 - Coefficients: 3 - Derivatives (3-order): 2.6460001468658447 - (x - a) ^ α: -0.045
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (2, 1, 2) - 1 / n!: 0.16666666666666666 - Coefficients: 3 - Derivatives (3-order): -1.05840003490448 - (x - a) ^ α: -0.075
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (2, 1, 3) - 1 / n!: 0.16666666666666666 - Coefficients: 6 - Derivatives (3-order): 1.1340000629425049 - (x - a) ^ α: -0.105
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (2, 2, 0) - 1 / n!: 0.16666666666666666 - Coefficients: 3 - Derivatives (3-order): -1.5875999927520752 - (x - a) ^ α: -0.05
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (2, 2, 1) - 1 / n!: 0.16666666666666666 - Coefficients: 3 - Derivatives (3-order): -1.05840003490448 - (x - a) ^ α: -0.075
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (2, 2, 2) - 1 / n!: 0.16666666666666666 - Coefficients: 1 - Derivatives (3-order): 0.10583999007940292 - (x - a) ^ α: -0.125
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (2, 2, 3) - 1 / n!: 0.16666666666666666 - Coefficients: 3 - Derivatives (3-order): -0.22679999470710754 - (x - a) ^ α: -0.175
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (2, 3, 0) - 1 / n!: 0.16666666666666666 - Coefficients: 6 - Derivatives (3-order): 1.7010000944137573 - (x - a) ^ α: -0.06999999999999999
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (2, 3, 1) - 1 / n!: 0.16666666666666666 - Coefficients: 6 - Derivatives (3-order): 1.1340000629425049 - (x - a) ^ α: -0.105
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (2, 3, 2) - 1 / n!: 0.16666666666666666 - Coefficients: 3 - Derivatives (3-order): -0.22679999470710754 - (x - a) ^ α: -0.175
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (2, 3, 3) - 1 / n!: 0.16666666666666666 - Coefficients: 3 - Derivatives (3-order): 0.0 - (x - a) ^ α: -0.24499999999999997
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (3, 0, 0) - 1 / n!: 0.16666666666666666 - Coefficients: 3 - Derivatives (3-order): -3.0764999389648438 - (x - a) ^ α: -0.028000000000000004
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (3, 0, 1) - 1 / n!: 0.16666666666666666 - Coefficients: 6 - Derivatives (3-order): -6.00600004196167 - (x - a) ^ α: -0.041999999999999996
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (3, 0, 2) - 1 / n!: 0.16666666666666666 - Coefficients: 6 - Derivatives (3-order): 1.7009999752044678 - (x - a) ^ α: -0.06999999999999999
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (3, 0, 3) - 1 / n!: 0.16666666666666666 - Coefficients: 3 - Derivatives (3-order): 0.0 - (x - a) ^ α: -0.09799999999999999
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (3, 1, 0) - 1 / n!: 0.16666666666666666 - Coefficients: 6 - Derivatives (3-order): -6.00600004196167 - (x - a) ^ α: -0.042
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (3, 1, 1) - 1 / n!: 0.16666666666666666 - Coefficients: 3 - Derivatives (3-order): -1.8580000400543213 - (x - a) ^ α: -0.063
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (3, 1, 2) - 1 / n!: 0.16666666666666666 - Coefficients: 6 - Derivatives (3-order): 1.1340000629425049 - (x - a) ^ α: -0.105
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (3, 1, 3) - 1 / n!: 0.16666666666666666 - Coefficients: 3 - Derivatives (3-order): 0.0 - (x - a) ^ α: -0.14699999999999996
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (3, 2, 0) - 1 / n!: 0.16666666666666666 - Coefficients: 6 - Derivatives (3-order): 1.7010000944137573 - (x - a) ^ α: -0.06999999999999999
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (3, 2, 1) - 1 / n!: 0.16666666666666666 - Coefficients: 6 - Derivatives (3-order): 1.1340000629425049 - (x - a) ^ α: -0.105
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (3, 2, 2) - 1 / n!: 0.16666666666666666 - Coefficients: 3 - Derivatives (3-order): -0.22679999470710754 - (x - a) ^ α: -0.175
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (3, 2, 3) - 1 / n!: 0.16666666666666666 - Coefficients: 3 - Derivatives (3-order): 0.0 - (x - a) ^ α: -0.24499999999999997
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (3, 3, 0) - 1 / n!: 0.16666666666666666 - Coefficients: 3 - Derivatives (3-order): 0.0 - (x - a) ^ α: -0.09799999999999999
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (3, 3, 1) - 1 / n!: 0.16666666666666666 - Coefficients: 3 - Derivatives (3-order): 0.0 - (x - a) ^ α: -0.14699999999999996
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (3, 3, 2) - 1 / n!: 0.16666666666666666 - Coefficients: 3 - Derivatives (3-order): 0.0 - (x - a) ^ α: -0.24499999999999997
    
    (x: 0, y: 1, z: 2, t: 3) - Variable Indices: (3, 3, 3) - 1 / n!: 0.16666666666666666 - Coefficients: 1 - Derivatives (3-order): 0.0 - (x - a) ^ α: -0.3429999999999999
    


As an example, we can obtain the components of the third-order Taylor series expansion of a four-variable function value at $(0, 0, 0, 0)$ around the point $(0.2, 0.3, 0.5, 0.7)$ above, where each derivative and coefficient of corresponding terms are displayed accordingly. That is,

$$
\begin{aligned}
&\text{(x: 0, y: 1, z: 2, t: 3)}, \newline
\newline
&\text{Variable Indices: } (0, 0), \newline
\newline
&\text{1 / n!: } 0.5, \newline
\newline
&\text{Coefficients: } 1.0, \newline
\newline
&\text{Derivatives (2-order): } -2.153550386428833, \newline
\newline
&\text{(x - a)}^{\alpha}: 0.04000000000000001
\end{aligned}
$$

where at the top variables are indexed to keep track of when taking derivatives; "variable indices" shows partial derivative orders with respect to variables; "coefficients" corresponds to multinomial coefficients; and $(x - a)^{\alpha}$ designates $(x_{1} - a_{1})^{\alpha_{1}} \cdots (x_{n} - a_{n})^{\alpha_{n}}$ multi-index notation for polynomial factors of variables. Therefore, we have $f_{xx}$ at $(0.2, 0.3, 0.5, 0.7) = -2.153550386428833$, and $(x - a) ^ n = (0 - 0.2)^{2} \approx 0.04000000000000001$.

```python
ftrue = fv(0.0, 0.0, 0.0, 0.0)

for i in range(6):
  print(f"True: {ftrue} - 'taylor_v1' approximate: {taylor_v1(fv, [0.0, 0.0, 0.0, 0.0], [0.2, 0.3, 0.5, 0.7], i):.6f}",
        f"- 'taylor_v2' approximate: {taylor_v2(fv, [xv, xv, xv, xv], [av, bv, cv, dv], i):.6f}",
        f"- 'taylor_v3' approximate: {taylor_v3(fi, jnp.array([0.0, 0.0, 0.0, 0.0]), jnp.array([0.2, 0.3, 0.5, 0.7]), i):.6f}",
        f"- 'taylor_jax' approximate: {taylor_jax(fv, [0.0, 0.0, 0.0, 0.0], [0.2, 0.3, 0.5, 0.7], i):.6f}",
        end='\n\n')
```

    True: 0.0 - 'taylor_v1' approximate: -0.054047 - 'taylor_v2' approximate: -0.054047 - 'taylor_v3' approximate: -0.054047 - 'taylor_jax' approximate: -0.054047
    
    True: 0.0 - 'taylor_v1' approximate: 0.166845 - 'taylor_v2' approximate: 0.166845 - 'taylor_v3' approximate: 0.166845 - 'taylor_jax' approximate: 0.166845
    
    True: 0.0 - 'taylor_v1' approximate: -0.135555 - 'taylor_v2' approximate: -0.135555 - 'taylor_v3' approximate: -0.135555 - 'taylor_jax' approximate: -0.135555
    
    True: 0.0 - 'taylor_v1' approximate: -0.036295 - 'taylor_v2' approximate: -0.036295 - 'taylor_v3' approximate: -0.036295 - 'taylor_jax' approximate: -0.036295
    
    True: 0.0 - 'taylor_v1' approximate: 0.066675 - 'taylor_v2' approximate: 0.066675 - 'taylor_v3' approximate: 0.066675 - 'taylor_jax' approximate: 0.066675
    
    True: 0.0 - 'taylor_v1' approximate: 0.005607 - 'taylor_v2' approximate: 0.005607 - 'taylor_v3' approximate: 0.005607 - 'taylor_jax' approximate: 0.005607
    


Taylor series approximations of three different Taylor functions (`taylor_v1`, `taylor_v2`, `taylor_v3`, `taylor_jax`) defined were compared above up to the sixth-order in terms of true and approximated function values. Results show exact agreement with each other. The higher the order of derivatives that approximation contains, the better the approximation gets to true value. This is exactly what is expected to happen with Taylor series in approximating a function in the vicinity of a point.

```python
results = {"func": [], "order": [], "dim": [], "duration": []}
functions = [multi_indices_v1,
             multi_indices_v2]
order_inputs = range(1, 9)
dim_inputs = range(1, 9)

for func, dim, order, in product(functions, order_inputs, dim_inputs):
  start_time = time.time()
  _ = [i for i in func(order, dim)]
  end_time = time.time()
  results["func"].append(func.__name__)
  results["order"].append(order)
  results["dim"].append(dim)
  results["duration"].append(end_time - start_time)

results = pd.DataFrame(results)

colors_list = ["red", "green", "blue", "gray", "darkorange", "cyan", "lime", "fuchsia"]

fig, ax = plt.subplots()
sns.lineplot(data=results,
             ax=ax,
             palette=colors_list,
             x="dim",
             y="duration",
             hue="order",
             style="func",
             markers=True,
             markersize=7,
             dashes=False)
ax.set_xlabel("Dimension", labelpad=15)
ax.set_ylabel("Time (s)", labelpad=40, rotation="horizontal")

handles, labels = ax.get_legend_handles_labels()
new_labels = []
for label in labels:
  if label == "order":
    new_labels.append("Order")
  elif label == "func":
    new_labels.append("Function")
  else:
    new_labels.append(label)
ax.legend(handles, new_labels, loc="best", edgecolor="black",
          fancybox=False, shadow=True, borderaxespad=1)

plt.tight_layout()
plt.show()
```


    
![Figure](codes/python/python_7_files/output_14_0.png)
    


The plot above indicates the execution time up to the eighth-order of both `multi_indices_v1` and `multi_indices_v2` functions that are to be used to obtain variable indices so as to take their respective partial derivatives. As small execution time means less overhead cost, `multi_indices_v2` approach is strikingly superior against the other in terms of speed, which is especially seen with data points for both higher orders and dimensions seen on the plot. If both functions are examined, the reason for such fast execution lies in the omission of double counts of indices (i.e., only $(0, 0, 1)$ instead of $(0, 0, 1)$, $(0, 1, 0)$, $(1, 0, 0)$). Therefore, the time complexity of `multi_indices_v1` grows exponentially.

```python
results = {"func": [], 'a': [], 'n': [], "duration": []}
functions = [multinomials_jax_v1,
             multinomials_jax_v2,
             multinomials_jax_v3,
             multinomials_jax_v4]
a_inputs = [jnp.array([0]),
            jnp.array([0, 1]),
            jnp.array([0, 1, 2]),
            jnp.array([0, 1, 2, 3]),
            jnp.array([0, 1, 2, 3, 4])]
n_inputs = range(1, 9)

for func, a, n, in product(functions, a_inputs, n_inputs):
  start_time = time.time()
  func(a, n).ravel()
  end_time = time.time()
  results["func"].append(func.__name__)
  results['a'].append(a.shape[0])
  results['n'].append(n)
  results["duration"].append(end_time - start_time)

results = pd.DataFrame(results)

colors_list = ["red", "green", "blue", "gray", "darkorange", "cyan", "lime", "fuchsia"]

fig, ax = plt.subplots()
sns.lineplot(data=results,
             ax=ax,
             palette=colors_list,
             x='a',
             y="duration",
             hue='n',
             style="func",
             markers=True,
             markersize=7,
             dashes=False)
ax.set_xticks(range(1, 6))
ax.set_xlabel("Dimension of input array (a)", labelpad=15)
ax.set_ylabel("Time (s)", labelpad=40, rotation="horizontal")

handles, labels = ax.get_legend_handles_labels()
new_labels = []
for label in labels:
  if label == 'n':
    new_labels.append("Order")
  elif label == "func":
    new_labels.append("Function")
  else:
    new_labels.append(label)
ax.legend(handles, new_labels, loc="best", edgecolor="black",
          fancybox=False, shadow=True, borderaxespad=1)

plt.tight_layout()
plt.show()
```


    
![Figure](codes/python/python_7_files/output_16_0.png)
    


Multinomial coefficients can be calculated by repeatedly applying outer product on a given vector `a` as defined by functions (`multinomials_jax_v1`, `multinomials_jax_v2`, `multinomials_jax_v3`, `multinomials_jax_v4`). The execution times of calculating multinomial coefficients based on four different functions defined at the beginning are depicted on the plot with respect to size of input array and order of derivative. Compared to alternatives, `multinomials_jax_v4` performs poorly unlike other three functions (`multinomials_jax_v1`, `multinomials_jax_v2`, `multinomials_jax_v3`) on account of lacking the calculation efficiency provided by **JAX** library. This becomes noticeable at the order of $6$ where the execution time of `multinomials_jax_v4` begins to diverge from others. Additionally, there is no visible performance discrepancy between the three in terms of both input array size and order of derivative.

```python
results = {"func": [], "order": [], "duration": []}
order_inputs = range(1, 6)

for o in order_inputs:
  start_time = time.time()
  _ = taylor_v1(fv, [0.0, 0.0, 0.0, 0.0], [0.2, 0.3, 0.5, 0.7], o)
  end_time = time.time()
  results["func"].append("taylor_v1")
  results["order"].append(o)
  results["duration"].append(end_time - start_time)
  start_time = time.time()
  start_time = time.time()
  _ = taylor_v2(fv, [xv, xv, xv, xv], [av, bv, cv, dv], o)
  end_time = time.time()
  results["func"].append("taylor_v2")
  results["order"].append(o)
  results["duration"].append(end_time - start_time)
  start_time = time.time()
  _ = taylor_v3(fi, jnp.array([0.0, 0.0, 0.0, 0.0]), jnp.array([0.2, 0.3, 0.5, 0.7]), o)
  end_time = time.time()
  results["func"].append("taylor_v3")
  results["order"].append(o)
  results["duration"].append(end_time - start_time)
  _ = taylor_jax(fv, [0.0, 0.0, 0.0, 0.0], [0.2, 0.3, 0.5, 0.7], o)
  end_time = time.time()
  results["func"].append("taylor_jax")
  results["order"].append(o)
  results["duration"].append(end_time - start_time)

results = pd.DataFrame(results)

colors_list = ["red", "blue", "green", "fuchsia"]

fig, ax = plt.subplots()
sns.lineplot(data=results,
             ax=ax,
             palette=colors_list,
             x="order",
             y="duration",
             hue="func",
             style="func",
             markers=True,
             markersize=7,
             dashes=False)
ax.set_xticks(range(1, 6))
ax.set_xlabel("Order of derivatives", labelpad=15)
ax.set_ylabel("Time (s)", labelpad=40, rotation="horizontal")
ax.legend(title='', loc="best", edgecolor="black",
          fancybox=False, shadow=True, borderaxespad=1)

plt.tight_layout()
plt.show()
```


    
![Figure](codes/python/python_7_files/output_18_0.png)
    


# Conclusion

`taylor_v3` is noticeably faster than the other three as demonstrated in the plot above. However, despite the fact that `taylor_v1` and `taylor_v2` have the same function generating indices (`multi_indices_v2`), the execution time of `taylor_v2` takes more than twice as long as `taylor_v1` does at the fourth- and fifth-order derivatives as can be seen through visual inspection in the plot above.
