# Linear Systems and Finite Difference Stencils in NumPy

```python
from itertools import product
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import timeit
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

It is known from the theory of linear algebra that solution to a linear system in the form of $\mathbf{A} \mathbf{x} = \mathbf{b}$ is a column vector ($\mathbf{x}$). We can solve a system of linear equations in **NumPy** in two ways with the help of `solve` function provided in `linalg` package shown below. The only difference between the two comes from the shape of $\mathbf{b}$ vector (i.e., the first shape is `(3,)` while the second is `(3, 1)`).

```python
A = np.array([[1, 3, -2], [3, 5, 6], [2, 4, 3]])
b = np.array([5, 7, 8])

print(f"'A' matrix:\n\n{A}",
      f"'b' vector:\n\n{b}",
      f"Solution of system:\n\n{np.linalg.solve(A, b)}",
      sep="\n\n",
      end="\n\n")

b = b[:, np.newaxis]

print(f"'A' matrix:\n\n{A}",
      f"'b' vector:\n\n{b}",
      f"Solution of system:\n\n{np.linalg.solve(A, b)}",
      sep="\n\n")
```

    'A' matrix:
    
    [[ 1  3 -2]
     [ 3  5  6]
     [ 2  4  3]]
    
    'b' vector:
    
    [5 7 8]
    
    Solution of system:
    
    [-15.   8.   2.]
    
    'A' matrix:
    
    [[ 1  3 -2]
     [ 3  5  6]
     [ 2  4  3]]
    
    'b' vector:
    
    [[5]
     [7]
     [8]]
    
    Solution of system:
    
    [[-15.]
     [  8.]
     [  2.]]


At this point a question can be raised about the applicability of solving multiple systems at once without using the Python for-loop. The answer to that is yes, and this method can be demonstrated below. The implementation of the method simply relies on combining each system ($\mathbf{A}\_{1}, \mathbf{A}\_{2}, \dots$ and $\mathbf{b}\_{1}, \mathbf{b}\_{2}, \dots$) with the help of **NumPy**'s function `stack` in a single array and solving it. We can treat $\mathbf{b}_{i}$ vectors as an array of shape either $(N,)$ or $(N, 1)$ in order to obtain the solution.

```python
A = np.array([[1, 3, -2], [3, 5, 6], [2, 4, 3]])
b = np.array([5, 7, 8])
A_stack = np.stack([A, A, A], axis=0)
b_stack = np.stack([b, b, b], axis=0)

print(f"'A' {A.shape}:\n\n{A}",
      f"'b' {b.shape}:\n\n{b}",
      f"'A_stack' {A_stack.shape}:\n\n{A_stack}",
      f"'b_stack' {b_stack.shape}:\n\n{b_stack}",
      f"Solutions of all systems by 'np.linalg.solve(A_stack, b_stack.T)':\n\n{np.linalg.solve(A_stack, b_stack.T)}",
      sep="\n\n",
      end="\n\n")

b = b[:, np.newaxis]
b_stack = np.stack([b, b, b], axis=0)

print(f"'b' {b.shape}:\n\n{b}",
      f"'b_stack' {b_stack.shape}:\n\n{b_stack}",
      f"Solutions of all systems by 'np.linalg.solve(A_stack, b_stack)':\n\n{np.linalg.solve(A_stack, b_stack)}",
      sep="\n\n")
```

    'A' (3, 3):
    
    [[ 1  3 -2]
     [ 3  5  6]
     [ 2  4  3]]
    
    'b' (3,):
    
    [5 7 8]
    
    'A_stack' (3, 3, 3):
    
    [[[ 1  3 -2]
      [ 3  5  6]
      [ 2  4  3]]
    
     [[ 1  3 -2]
      [ 3  5  6]
      [ 2  4  3]]
    
     [[ 1  3 -2]
      [ 3  5  6]
      [ 2  4  3]]]
    
    'b_stack' (3, 3):
    
    [[5 7 8]
     [5 7 8]
     [5 7 8]]
    
    Solutions of all systems by 'np.linalg.solve(A_stack, b_stack.T)':
    
    [[[-15. -15. -15.]
      [  8.   8.   8.]
      [  2.   2.   2.]]
    
     [[-15. -15. -15.]
      [  8.   8.   8.]
      [  2.   2.   2.]]
    
     [[-15. -15. -15.]
      [  8.   8.   8.]
      [  2.   2.   2.]]]
    
    'b' (3, 1):
    
    [[5]
     [7]
     [8]]
    
    'b_stack' (3, 3, 1):
    
    [[[5]
      [7]
      [8]]
    
     [[5]
      [7]
      [8]]
    
     [[5]
      [7]
      [8]]]
    
    Solutions of all systems by 'np.linalg.solve(A_stack, b_stack)':
    
    [[[-15.]
      [  8.]
      [  2.]]
    
     [[-15.]
      [  8.]
      [  2.]]
    
     [[-15.]
      [  8.]
      [  2.]]]


While the former generates a solution array (not a vector) filled with elements of solution vectors reduntantly:

```
 [[[-15. -15. -15.]
   [  8.   8.   8.]
   [  2.   2.   2.]]

  [[-15. -15. -15.]
   [  8.   8.   8.]
   [  2.   2.   2.]]

  [[-15. -15. -15.]
   [  8.   8.   8.]
   [  2.   2.   2.]]]
```

Therefore, it is reasonable to use the latter since it gives solution vectors column-wise on top of each other:

```
 [[[-15.]
   [  8.]
   [  2.]]

  [[-15.]
   [  8.]
   [  2.]]

  [[-15.]
   [  8.]
   [  2.]]]
```

Finite difference method is one of the fundamental approaches utilized in the solution of differential equations. It is used to approximate derivative of a function as a linear combination of function values at certain points. That is,

$$
f^{(d)}(x_{0}) \approx \sum_{i} a_{i} f(x_{0} + s_{i} h)
$$

where $a_{i}$ is finite difference coefficient, $s_{i}$ is the stencil point that can be an integer or real number, and $h$ is a small, positive, real number. Each stencil is simply a set of offsets from the evaluation point ($x_{0}$). In total, there are three stencil schemes, (e.g., backward, central, and forward). There is an [algorithm](https://en.wikipedia.org/wiki/Finite_difference_coefficient#Arbitrary_stencil_points) to obtain finite difference coefficients that involves solving a system of linear equations. For given stencil points and order of derivative, we solve the linear system

$$
\begin{pmatrix}
s_{1}^{0} & \cdots & s_{N}^{0} \newline
\vdots & \ddots & \vdots \newline
s_{1}^{N-1} & \cdots & s_{N}^{N-1}
\end{pmatrix}
\begin{pmatrix}
a_{1} \newline
\vdots \newline
a_{N}
\end{pmatrix}
=d!
\begin{pmatrix}
\delta_{0,d} \newline
\vdots \newline
\delta_{i,d} \newline
\vdots \newline
\delta_{N-1,d}
\end{pmatrix}
$$

where $s_{i}^{j}$ terms are elements of transposed [Vandermonde matrix](https://en.wikipedia.org/wiki/Vandermonde_matrix) generated by $N$ number of stencil points; $a_{i}$ terms are finite difference coefficients; $\delta_{i,j}$ is [Kronecker delta](https://en.wikipedia.org/wiki/Kronecker_delta) function; and $d$ is the order of derivative.

Now that a relevant question arises: given a list of stencil points and order of derivatives, how can we obtain those coefficients? This can be achieved by iteratively solving each linear system pairwise (a pair of stencil points and order of derivative) as defined by `stencil_coeffs` function below. However, it should be noted that this "solve-at-once" approach only works for the same number of stencil points because `stack` operation of **NumPy** is applied to arrays of compatible shapes.

```python
def stencil_coeffs(stencils, orders):
  V = []
  K = []
  for s, d in zip(stencils, orders):
    vanmat = np.vander(s, increasing=True).T
    krondel = np.zeros(s.shape)
    krondel[d] = np.prod(np.arange(1, d + 1))
    K.append(krondel[:, np.newaxis])
    V.append(vanmat)
  return np.linalg.solve(np.stack(V), np.stack(K))
```

```python
stencils = [np.array([-1, 0, 1, 2, 3]),
            np.array([0, 1, 2, 3, 4]),
            np.array([-2, -1, 0, 1, 2]),
            np.array([-3, -2, -1, 0, 1])]
orders = [1, 2, 3, 4]
result = stencil_coeffs(stencils, orders)

print(f"'1st derivative' - 'Stencil': {stencils[0]} - Finite difference coefficients: {result[0].squeeze()}",
      f"'2nd derivative' - 'Stencil': {stencils[1]} - Finite difference coefficients: {result[1].squeeze()}",
      f"'3rd derivative' - 'Stencil': {stencils[2]} - Finite difference coefficients: {result[2].squeeze()}",
      f"'4th derivative' - 'Stencil': {stencils[3]} - Finite difference coefficients: {result[3].squeeze()}",
      sep="\n\n")
```

    '1st derivative' - 'Stencil': [-1  0  1  2  3] - Finite difference coefficients: [-0.25       -0.83333333  1.5        -0.5         0.08333333]
    
    '2nd derivative' - 'Stencil': [0 1 2 3 4] - Finite difference coefficients: [ 2.91666667 -8.66666667  9.5        -4.66666667  0.91666667]
    
    '3rd derivative' - 'Stencil': [-2 -1  0  1  2] - Finite difference coefficients: [-0.5  1.   0.  -1.   0.5]
    
    '4th derivative' - 'Stencil': [-3 -2 -1  0  1] - Finite difference coefficients: [ 1. -4.  6. -4.  1.]


```python
repeat = 25
loop = 100

results = {"type": [], "order": [], "accuracy": [], "dim": [], "duration": []}
def backward_stencil(n): return np.arange(-2 * n, 1)
def central_stencil(n): return np.arange(-n, n + 1)
def forward_stencil(n): return np.arange(0, 2 * n + 1)
n_inputs = range(1, 6)
dim_inputs = range(1, 5)

for func, n, dim in product([backward_stencil, central_stencil, forward_stencil], n_inputs, dim_inputs):
  for d in range(1, n + 1):
    stencil = func(n)
    t = timeit.Timer(lambda: stencil_coeffs([stencil] * dim, [d] * dim))
    total_time = min(t.repeat(repeat=repeat, number=loop))
    avg_time = total_time / loop
    results["type"].append(func.__name__)
    results["order"].append(d)
    if func.__name__ == "central_stencil":
      results["accuracy"].append(stencil.shape[0] + 1 - 2 * ((d + 1) // 2))
    else:
      results["accuracy"].append(stencil.shape[0] - d)
    results["dim"].append(dim)
    results["duration"].append(avg_time)

results = pd.DataFrame(results)
results["type"] = results["type"].astype("category")

fig, axes = plt.subplots(1, 3, figsize=(14, 6), sharey=True)

for i, kind in enumerate(["backward_stencil", "central_stencil", "forward_stencil"]):
  df = results[results["type"] == kind]
  if kind == "central_stencil":
    color_num = 5
  else:
    color_num = 9
  sns.lineplot(data=df,
               ax=axes[i],
               palette=sns.color_palette("bright", color_num),
               x="order",
               y="duration",
               hue="accuracy",
               style="dim",
               markers=True,
               markersize=7,
               dashes=False)
  axes[i].set_xticks(df["order"].sort_values().unique())
  axes[i].set_xlabel("Order of derivative", labelpad=15)
  axes[i].set_ylabel('')
  axes[i].set_title(kind.title().replace('_', ' '), fontsize=14, pad=15)
  leg = axes[i].legend(loc="lower left", mode="expand", edgecolor="black",
                 fancybox=False, shadow=True,
                 bbox_to_anchor=(0, 1.175, 1, 0.102), ncols=4, borderaxespad=0,
                 handlelength=1.25)
axes[0].set_yscale("log")
axes[0].set_ylabel("Time (s)", labelpad=40, rotation="horizontal")

plt.tight_layout()
plt.show()
```


    
![Figure](assets/codes/python/python_9_files/output_11_0.png)
    


A DataFrame above was created with respect to different combinations of parameters (i.e., number of stencil points, order of derivatives, and number of dimensions). `n_inputs` list numbered from $1$ to $5$ is a collection of number of candidate stencil points to use for backward, central, and forward stencil schemes for `backward_stencil`, `central_stencil`, and `forward_stencil` functions that generate equal number of stencil points for a given `n`. `dim_inputs` variable stores number of dimensions to compute those coefficients at once, which ranges from $1$ to $4$. A rough estimate of the order of truncation error is related to number of stencil points (`N`) and order of derivative (`d`). For forward and backward schemes it becomes $N - d$, and for central scheme it is $N + 1 - 2 \left\lfloor \frac{d + 1}{2} \right\rfloor$. And the results were plotted against those combinations.

# Conclusion

In this study a performance analysis was conducted on how fast the function that generates finite difference coefficients with respect to a variety of accuracy degrees as shown in different colors and orders of derivative varying along the x-axis. In other words, how performance scales with problem size. The result shows that even under the most extreme conditions (i.e., higher order of derivatives and accuracy degrees), the execution times level at around $10^{-4}$ seconds.
