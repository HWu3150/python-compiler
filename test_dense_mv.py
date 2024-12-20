import numpy as np
from numba import njit

from get_ssa_compiler import ssa_by_blocks, blocks, GetSSACompiler
from graph_viz import viz_ast_and_cfg


# dense matrix vector multiplication
@njit(pipeline_class=GetSSACompiler)
def dense_mv(A, x):
    """
    Perform a dense matrix-vector multiplication.

    Args:
        A: A numpy array representing the matrix.
        x: A numpy array representing the vector.

    Returns:
        The result of the matrix-vector multiplication.
    """
    y = np.zeros(A.shape[0])
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            y[i] = A[i, j] * x[j]
    return y


ssa_by_blocks.clear()
blocks.clear()
func_ir = None

dense_mv(np.zeros((2, 2)), np.zeros(2))
print("SSA Statements Grouped by Block:")
for blk_offset, ssa_list in ssa_by_blocks.items():
    print(f"Block {blk_offset}:")
    for stmt in ssa_list:
        print(f"  {stmt}")
print()

viz_ast_and_cfg(blocks, dense_mv)