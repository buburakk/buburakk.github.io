# Computing Pascal's Triangle Efficiently

```python
from itertools import product
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import timeit
import sys
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
print(f"Maximum Python integer value: {sys.maxsize}",
      f"Maximum NumPy unsigned integer value: {np.iinfo(np.uint64).max}",
      sep="\n\n")
```

    Maximum Python integer value: 9223372036854775807
    
    Maximum NumPy unsigned integer value: 18446744073709551615


Above we first obtain the maximum integer values representable for Python's built-in integers and **NumPy**'s unsigned 64-bit integers to determine machine limit for overflow when calculating [Pascal's triangle](https://en.wikipedia.org/wiki/Pascal%27s_triangle) numbers for larger rows. It can be noted that **NumPy** offers higher computation capacity with its unsigned integer data type.

```python
def pascal_triangle(row_number):
  row = [1]
  for r in range(row_number - 1):
    stop_index = r // 2
    next_row = [1]
    for i in range(1, (stop_index) + 1):
      next_row.append(row[i - 1] + row[i])
    if r % 2:
      row = next_row + [2 * row[stop_index]] + next_row[::-1]
    if not r % 2:
      row = next_row + next_row[::-1]
  return row


def pastri_v1(row_number):
  row = [1]
  for _ in range(1, row_number):
    next_row = [1]
    for i in range(len(row) - 1):
      next_row.append(row[i] + row[i + 1])
    next_row.append(1)
    row = next_row
  return row


def pastri_v2(row_number):
  if row_number <= 1:
    return [1]
  prev_row = pastri_v2(row_number - 1)
  curr_row = [1]
  for i in range(len(prev_row) - 1):
    curr_row.append(prev_row[i] + prev_row[i + 1])
  curr_row.append(1)
  return curr_row


def pastri_np_v1(row_number):
  row = np.array([1], dtype=np.uint64)
  for _ in range(1, row_number):
    row = np.pad(row, (1, 1), constant_values=0)
    row = row[:-1] + row[1:]
  return row


def pastri_np_v2(row_number):
  if row_number <= 1:
    return np.array([1], dtype=np.uint64)
  prev_row = pastri_np_v2(row_number - 1)
  prev_row = np.pad(prev_row, (1, 1), constant_values=0)
  return prev_row[:-1] + prev_row[1:]
```

While `pastri_v2` and `pastri_np_v2` functions are recursive implementations of generating Pascal's triangle numbers for a given row, `pascal_triangle`, `pastri_v1`, `pastri_np_v1` functions are iterative implementations. Hence, the recursive functions are bounded by a certain number of recursion depth unlike the iterative functions. Additionally, the function (`pastri_np_v1`) implemented through **NumPy** arrays allows to obtain Pascal's triangle numbers for larger row numbers due to `uint64` data type that enables to represent larger numbers without overflow.

```python
for n in range(1, 101):
  if sys.maxsize < max(pascal_triangle(n)):
    print(f"Maximum input size to use without overflow: {n}")
    break
```

    Maximum input size to use without overflow: 68


Based on the smaller value of machine limit, we obtain the maximum row number to calculate corresponding numbers within Pascal's triangle prior to overflowing, which is $68$ as determined above.

```python
repeat = 5
loop = 100

results = {"func": [], "size": [], "duration": []}
functions = [pascal_triangle, pastri_v1, pastri_v2, pastri_np_v1, pastri_np_v2]
size_inputs = [5, 15, 25, 35, 45, 55, 65]

for func, size in product(functions, size_inputs):
  t = timeit.Timer(lambda: func(size))
  total_time = min(t.repeat(repeat=repeat, number=loop))
  avg_time = total_time / loop
  results["func"].append(func.__name__)
  results["size"].append(size)
  results["duration"].append(avg_time)

results = pd.DataFrame(results)
results["func"] = results["func"].astype("category")
```

In this step, the execution time of each function for different input sizes (row number in Pascal's triangle), $[5, 15, 25, 35, 45, 55, 65]$, was measured, the results data was stored the in a DataFrame, which was subsequently used for visualization.

```python
fig, ax = plt.subplots()
sns.lineplot(data=results, ax=ax, x="size", y="duration", hue="func", style="func", markers=True, markersize=7, dashes=False)

plt.xticks(size_inputs)
plt.xlabel("Input size", labelpad=15)
plt.yscale("log")
plt.ylabel("Time (s)", labelpad=40, rotation="horizontal")
plt.legend(loc="best", edgecolor="black", fancybox=False, shadow=True, borderaxespad=1)
plt.tight_layout()
plt.show()
```


    
![Figure](assets/codes/python/python_8_files/output_11_0.png)
    


# Conclusion

This notebook compares the performance of five different functions for generating a row of numbers from Pascal's triangle: `pascal_triangle`, `pastri_v1`, `pastri_v2`, `pastri_np_v1`, and `pastri_np_v2`. Based on the plot, the execution times of `pascal_triangle`, `pastri_v1`, and `pastri_v2` functions (using standard Python lists) diverge from `pastri_np_v1` and `pastri_np_v2` functions (using **NumPy** arrays) in performance offering shorter execution time. This suggests that for the range of input sizes tested, the overhead introduced by using **NumPy** for these specific calculations outweighs the potential benefits of **NumPy**'s vectorized operations. Also the maximum input size tested is limited by Python's environment for functions `pascal_triangle`, `pastri_v1`, `pastri_v2` defined in it. For significantly larger input sizes where integer overflow becomes a major concern for the list-based implementations, the **NumPy** implementations (especially with `uint64` as used here) might be of use or even necessary.
