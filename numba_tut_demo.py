import numba
from numba import njit, jit, vectorize, prange

import time
import numpy as np

######################################################################################################
# Demo: performance of using Numba vs. not using it
######################################################################################################

@njit
def sum_of_squares(arr):
    ss = 0.0
    for i in arr:
        ss += i ** 2
    return ss

arr = np.random.rand(10**6)

start = time.time()
result_python = sum(i ** 2 for i in arr)
end = time.time()
print(f"Without Numba: Result={result_python}, Time={end - start:.6f} seconds")

start = time.time()
result_numba = sum_of_squares(arr)
end = time.time()
print(f"With Numba: Result={result_numba}, Time={end - start:.6f} seconds")

start = time.time()
result_numba = sum_of_squares(arr)
end = time.time()
print(f"With Numba (second run): Result={result_numba}, Time={end - start:.6f} seconds")

######################################################################################################
# Demo: nopython vs object mode
######################################################################################################

class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def future_age(self, years):
        return self.age + years

@njit
def calculate_future_age():
    person = Person("Jake", 30)
    return person.future_age(10)

print(calculate_future_age()) # an error is expected
