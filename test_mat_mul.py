import numpy as np
from numba import njit

from get_ssa_compiler import ssa_by_blocks, blocks, GetSSACompiler
from graph_viz import viz_ast_and_cfg


@njit(pipeline_class=GetSSACompiler)
def matmul(A, B):
    """
    Perform a matrix multiplication.

    Args:
        A: A numpy array representing the first matrix.
        B: A numpy array representing the second matrix.

    Returns:
        The result of the matrix multiplication.
    """
    C = np.zeros((A.shape[0], B.shape[1]))
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            for k in range(A.shape[1]):
                C[i, j] += A[i, k] * B[k, j]
    return C

ssa_by_blocks.clear()
blocks.clear()
func_ir = None

matmul(np.zeros((2, 2)), np.zeros((2, 2)))
print("SSA Statements Grouped by Block:")
for blk_offset, ssa_list in ssa_by_blocks.items():
    print(f"Block {blk_offset}:")
    for stmt in ssa_list:
        print(f"  {stmt}")
print()

viz_ast_and_cfg(blocks, matmul)
