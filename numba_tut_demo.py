import numba
from numba import njit, jit, vectorize
from numba.core.compiler import CompilerBase, DefaultPassBuilder
from numba.core.compiler_machinery import FunctionPass, register_pass
from numba.core.errors import TypingError

import time
import numpy as np

######################################################################################################
# Demo: performance of using Numba vs. not using it
######################################################################################################
from numba.core.untyped_passes import IRProcessing

print("Demo: performance of using Numba vs. not using it")


@njit
def sum_of_squares(arr):
    ss = 0.0
    for i in arr:
        ss += i ** 2
    return ss


arr = np.random.rand(10 ** 6)

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
print(f"With Numba (second run): Result={result_numba}, Time={end - start:.6f} seconds\n")

######################################################################################################
# Demo: nopython vs object mode
######################################################################################################
print("Demo: nopython vs object mode")


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


try:
    print(calculate_future_age())
except TypingError as e:
    print("Caught Numba Typing Error")
    print(e)


@jit(forceobj=True)
def calculate_future_age2():
    person = Person("Jake", 30)
    return person.future_age(10)


print(calculate_future_age2(), "\n")

######################################################################################################
# Demo: Function Vectorization
######################################################################################################
print("Demo: Function Vectorization")


@vectorize(["float64(float64, float64)"], target="cpu")
def square_sum(x, y):
    return x ** 2 + y ** 2


arr1 = np.random.rand(10 ** 7)
arr2 = np.random.rand(10 ** 7)

start = time.time()
result_numpy = arr1 ** 2 + arr2 ** 2
end = time.time()
print(f"Without Numba: Time={end - start:.6f} seconds")

start = time.time()
result_numba = square_sum(arr1, arr2)
end = time.time()
print(f"With Numba Vectorize: Time={end - start:.6f} seconds")

start = time.time()
_ = square_sum(arr1, arr2)
end = time.time()
print(f"With Numba Vectorize (second run): Time={end - start:.6f} seconds")

print(f"Check results: {np.allclose(result_numpy, result_numba)}")

######################################################################################################
# Demo: Create customized compiler pass & customized compiler
######################################################################################################
print("Demo: Create customized compiler pass & customized compiler")


@register_pass(mutates_CFG=False, analysis_only=True)
class Test(FunctionPass):
    _name = "Test"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        return False


class TestCompiler(CompilerBase):

    def define_pipelines(self):
        pm = DefaultPassBuilder.define_nopython_pipeline(self.state)
        pm.add_pass_after(Test, IRProcessing)
        pm.finalize()

        print("Pipeline passes in PassManager:")
        for i, (pass_cls, _) in enumerate(pm.passes):
            print(f"{i + 1}. {pass_cls._name}")    # print each compiler pass
        return [pm]

@njit(pipeline_class=TestCompiler)
def test():
    return 5

_ = test()
