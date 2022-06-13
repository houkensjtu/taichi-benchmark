import taichi as ti
import time
import numpy as np
from numba import njit, prange

ti.init(arch=ti.cpu, default_fp=ti.f64)

n = 1 << 23
print('Problem size:', n)
v1 = ti.field(dtype=float, shape = n)
v2 = ti.field(dtype=float, shape = n)

@ti.kernel
def init():
    for i in range(n):
        v1[i] = 1.0
        v2[i] = 2.0

@ti.kernel
def reduce_ti():
    n = v1.shape[0]
    sum = 0.0
    ti.loop_config(block_dim=1024)
    for i in range(n):
        sum += v1[i]*v2[i]
    # return sum
    # Returning the sum in CUDA will slower the code. Why?


@njit(parallel=True, fastmath=True) # njit = jit (nopython=True)
def reduce_nb():
    sum = 0.0
    n = v1np.shape[0]
    for i in prange(n):
        sum += v1np[i] * v2np[i]


def reduce_np():
    return np.dot(v1np, v2np)

num_runs = 1000
print('Initializing...')
init()
v1np = v1.to_numpy()
v2np = v2.to_numpy()
reduce_ti() # Skip the first run to avoid compilation time
reduce_nb() # Skip the first run to avoid compilation time

print('Reducing in Taichi scope with a parallelized kernel...')
start = time.perf_counter()
for _ in range(num_runs):
    reduce_ti()
print(time.perf_counter() - start)

print('Reducing in Numba with @njit...')
start = time.perf_counter()
for _ in range(num_runs):
    reduce_nb()
print(time.perf_counter() - start)

print('Reducing in Numpy with dot...')
start = time.perf_counter()
for _ in range(num_runs):
    reduce_np()
print(time.perf_counter() - start)

