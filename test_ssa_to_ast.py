from numba import njit

from ast_visualize import visualize
from get_ssa_compiler import GetSSACompiler, ssa_statements


# Below are tested functions.

@njit(pipeline_class=GetSSACompiler)
def test_ast():
    a = 10
    a = a + 10
    c = a * 2
    return c

@njit(pipeline_class=GetSSACompiler)
def dce_test():
    a = 10
    b = 20
    c = 40
    d = c + a
    e = d + b
    c = a + b
    return c

test_ast()
visualize(test_ast, ssa_statements)

dce_test()
visualize(dce_test, ssa_statements)
