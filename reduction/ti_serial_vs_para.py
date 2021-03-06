# See related issue discussion on Github:
# https://github.com/taichi-dev/taichi/issues/4541
# Important conlusions:
# 1. Skip first kernel run to avoid compilation time
# 2. Tweak block_dim for cpu to maximize perf
# 3. Problem size needs to be large enough for para to outperform seri
# 4. Serialized loop with no return value will be optimized (must have a return value)

import taichi as ti
import time
import numpy as np
from numba import njit, prange

ti.init(arch=ti.cpu, default_fp=ti.f64)

n = 1 << 26
print('Problem size:', n)
v1 = ti.field(dtype=float, shape = n)
v2 = ti.field(dtype=float, shape = n)

@ti.kernel
def init():
    for i in range(n):
        v1[i] = 1.0
        v2[i] = 2.0

        
@ti.kernel
def reduce_ti_serial()->ti.f64:
    n = v1.shape[0]
    sum = 0.0
    ti.loop_config(serialize=True)
    for i in range(n):
        sum += v1[i] * v2[i]
    return sum


@ti.kernel
def reduce_ti_para()->ti.f64:
    n = v1.shape[0]
    sum = 0.0
    ti.loop_config(block_dim_adaptive=True)    
    for i in range(n):
        sum += v1[i] * v2[i]
    return sum
    

num_runs = 100
print('Initializing...')
init()
reduce_ti_serial() # Skip the first run to avoid compilation time
reduce_ti_para() # Skip the first run to avoid compilation time


print('Reducing in Taichi scope with a serialized loop...')
start = time.perf_counter()
for _ in range(num_runs):
    reduce_ti_serial()
print(time.perf_counter() - start)


print('Reducing in Taichi scope with a parallelized loop...')
start = time.perf_counter()
for _ in range(num_runs):
    reduce_ti_para()
print(time.perf_counter() - start)

