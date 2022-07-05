import taichi as ti
import time

ti.init(arch=ti.cpu)

n = 409600
v1 = ti.field(dtype=float, shape = n)
results = ti.field(dtype=float, shape = ())

@ti.kernel
def init():
    for i in range(n):
        v1[i] = 1.0

@ti.kernel
def reduce_ti_para():
    n = v1.shape[0]
    sum = 0.0
    for i in range(n):
        sum += v1[i]
    results[None] = sum


init()
reduce_ti_para() # Skip the first run to avoid compilation time

start = time.perf_counter()
for _ in range(1000):
    reduce_ti_para()
print(time.perf_counter() - start)

print(results[None])
