# Benchmarking Prime Number Generation: Naive Division vs. Optimized Methods

```python
from math import isqrt
import matplotlib as mpl
import matplotlib.pyplot as plt
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

```python
def prime_generator():
  primes_dict = dict()
  i = 1
  num = 2
  while True:
    if all(num % p != 0 for p in iter(primes_dict.values())):
      primes_dict[i] = num
      yield num
    i += 1
    num += 1


def prime_list_v1(num):
  prime_gen = prime_generator()
  return [next(prime_gen) for _ in range(num)]


def prime_list_v2(num):
  prime_check = [True] * num
  prime_check[0] = False
  prime_check[1] = False
  for i in range(2, isqrt(num) + 1):
    if prime_check[i]:
      for j in range(i * i, num, i):
        prime_check[j] = False
  return [i for i in range(num) if prime_check[i]]
```

```python
gen_primes = list()
eras_primes = list()

repeat = 5
loop = 5

for i in range(1, 6):
  t = timeit.Timer(lambda: prime_list_v1(5 ** i))
  total_time = min(t.repeat(repeat=repeat, number=loop))
  avg_time = total_time / loop
  gen_primes.append(avg_time)

  t = timeit.Timer(lambda: prime_list_v2(5 ** i))
  total_time = min(t.repeat(repeat=repeat, number=loop))
  avg_time = total_time / loop
  eras_primes.append(avg_time)

t = [5 ** i for i in range(1, 6)]

fig, ax = plt.subplots()

ax.plot(t, gen_primes, "r-", label="prime_list_v1")
ax.plot(t, eras_primes, "b--", label="prime_list_v2")
ax.set_xlabel("Size (n)", labelpad=10)
ax.set_yscale("log")
ax.set_ylabel("Time (s)", labelpad=40, rotation="horizontal")
ax.legend(loc="best", edgecolor="black",
          fancybox=False, shadow=True, borderaxespad=1)

plt.tight_layout()
plt.show()
```


    
![Figure](assets/codes/python/python_1_files/output_4_0.png)
    


# Conclusion

The code above runs both algorithms on a range of input sizes from $5$ to $3125$ and plots the execution time against the input size, comparing the performance of two prime number generation algorithms. This range was kept this narrow as it suffices to clearly show performance difference between the algorithms, which are simply explained below

* `prime_list_v1`: This function uses a simple algorithm to generate prime numbers by checking the next candidate successively against the existing list of primes.

* `prime_list_v2`: This algorithm was based on a variant of [Sieve of Eratosthenes](https://en.wikipedia.org/wiki/Sieve_of_Eratosthenes), which is a more efficient way to generate prime numbers by creating an array of all numbers up to a given limit and then marking all multiples of primes as non-prime.

The results show that the Sieve of Eratosthenes is a more efficient prime number generation algorithm than the prime generator function for larger input sizes. This is because the Sieve of Eratosthenes algorithm only needs to check each number once, while the prime generator function needs to check each number against the entire list of existing primes.
