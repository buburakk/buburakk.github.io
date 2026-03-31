# Array Compression: A Problem from Engineering

As I was working on my thesis, I came across some kind of inconvenience regarding results of optimization problems that I was solving. I had to arrange unidirectional fibers within each layer at such an angle that would give the optimal solution in design for maximum buckling resistance; and the number of layers can be either a few (e.g., four, eight) or as many as possible (e.g., thirty-two, sixty-four). Due to parametric nature of the design, there were many subcases of optimization problems to solve. Therefore, the results had to be filled in long tables. It entailed obtaining an array of numbers corresponding to angles of those fibers inside layers that might follow certain patterns. And there is a convention to simplify them into a more compact, readable form, which is named "laminate code" in composite design. A series of possible results are in array form as follows:

```
[45, 90, 0, 90, 0, 90, 90, -45, 0, -45, 0, 0, 0, 90, 90, 90],
[45, 90, 0, 90, 0, 0],
[-45, -45, -45, 0, 0, 0, 45],
[0, 0, 0, 90, 90, 0, 0, 90, 90, -45, 45],
[0, 0, -45, 45, -45, 45, 0, 90, 0, 90, 0, 0],
[15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15],
[45, 45, 45, 30, 30, 0, 30, 30, 45, 45, 45].
```

Each number the arrays is an orientation of fibers in layers. In the laminate code there are rules to make those simplifications as listed below.

1. Use opening ([) and closing (]) square brackets in order to designate the list of all angles at the beginning and end of it.

2. Use forward slash (/) to separate angles/patterns from each other.

3. Represent a consecutively repeated pattern with a subscripted number, which is number of times the pattern is repeated in the array, unless its occurrence is one. If it is a single angle, use only the subscript; otherwise, enclose the pattern with parentheses (e.g., $(30_{3} / 45 / 90_{9})$, $45_{7}$).

4. If there is a pattern with its mirror/reflective symmetry (i.e., this definition somewhat refers to palindrome except for the case where the length of pattern is an odd number) in the array, denote it with a subscript capital letter S (e.g., $(15 / 30 / 45_{4})_{S}$). The rule 3. is a degenerate case of the symmetry for even number of times a pattern is repeated, hence it must be ignored. For example, while "abcdcba" is a palindrome, but it is not mirror symmetric; and "abccba", on the other hand, is both mirror symmetric and a palindrome.

With the rules stipulated, the arrays above are converted into the following LaTeX-ready results:

```
[45 / (90 / 0)_{2} / 90_{2} / (-45 / 0)_{2} / 0_{2} / 90_{3}],
[45 / (90 / 0)_{2} / 0],
[-45_{3} / 0_{3} / 45],
[0 / (0_{2} / 90_{2})_{2} / -45 / 45],
[0_{2} / (-45 / 45)_{2} / (0 / 90)_{2} / 0_{2}],
[15_{15}],
[45_{3} / 30_{2} / 0 / 30_{2} / 45_{3}].
```

I have pondered a solution algorithm for quite a while, wondering whether it is possible to write an algorithm that would implement exactly aforementioned rules in order to obtain compact representations. Even though the steps to be taken were set out clearly, termination condition was still missing. When the raw array inputs and the formatted outputs are compared, there is a common point through which we can utilize for our halting condition for the algorithm. At some point it occurred to me that the point is the forward slash. By incorporating it into a criterion that chooses solution candidates of fewer number of forward slashes. Consequently, it turns out to be an optimization problem that we want to minimize the number of forward slashes for a given array. With this step we can complete the algorithm.

```python
def _to_expression(text, slashes):
    return {"text": text, "slashes": slashes}


def _to_literal(arr, i):
    return _to_expression(str(arr[i]), 0)


def _to_concatenate(left, right):
    return _to_expression(f"{left["text"]} / {right["text"]}",
                          left["slashes"] + right["slashes"] + 1)


def _is_repetition(arr, i, j, block_size):
    block = arr[i:i + block_size]
    for k in range(i, j + 1, block_size):
        if arr[k:k + block_size] != block:
            return False
    return True


def _is_symmetric(arr, i, j):
    length = j - i + 1
    if length % 2 != 0:
        return False
    half = length // 2
    return arr[i:i + half] == arr[i + half:j + 1][::-1]
```

The problem can be tackled through dynamic programming approach. To this end, we first define our helper functions above by which we specify our symmetric and repetition conditions (i.e., `_is_repetition`, `_is_symmetric`) along with utility tools that help us keep track of the conditions.

```python
from functools import cache


@cache
def _solver(arr, i, j):
    if i == j:
        return _to_literal(arr, i)
    candidates = []
    for k in range(i, j):
        left = _solver(arr, i, k)
        right = _solver(arr, k + 1, j)
        candidates.append(_to_concatenate(left, right))
    segment_size = j - i + 1
    repetition_found = False
    for block_size in range(1, segment_size // 2 + 1):
        if segment_size % block_size != 0:
            continue
        if _is_repetition(arr, i, j, block_size):
            base = _solver(arr, i, i + block_size - 1)
            count = segment_size // block_size
            if base["slashes"] == 0:
                text = f"{base["text"]}_{{{count}}}"
            else:
                text = f"({base["text"]})_{{{count}}}"
            candidates.append(_to_expression(text, base["slashes"]))
            repetition_found = True
            break
    if not repetition_found and _is_symmetric(arr, i, j):
        half = segment_size // 2
        base = _solver(arr, i, i + half - 1)
        if base["slashes"] == 0:
            text = f"{base["text"]}_{{S}}"
        else:
            text = f"({base["text"]})_{{S}}"
        candidates.append(_to_expression(text, base["slashes"]))
    return min(candidates, key=lambda e: (e["slashes"], len(e["text"])))


def laminate_code(arr):
    n = len(arr)
    arr = tuple(arr)
    return f"[{_solver(arr, 0, n - 1)["text"]}]"
```

The core of the algorithm relies on `_solver` function above that recursively searches for those patterns defined, and chooses the solution from among candidates by `min(candidates, key=lambda e: (e["slashes"], len(e["text"])))`. Moreover, `@cache` decorator at the top of function definition that takes advantage of memoization technique in order to speed up the algorithm by avoiding repeated calculations, if any, while it progresses.
