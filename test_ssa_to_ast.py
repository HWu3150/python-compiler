from numba import njit


@njit(pipeline_class=GetSSACompiler)
def test_ast():
  a = 10
  a = a + 10
  c = a * 2
  return c