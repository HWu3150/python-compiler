from numba import njit

from get_ssa_compiler import GetSSACompiler, ssa_by_blocks, blocks
from graph_viz import viz_ast_and_cfg

import numba

from recontruct_ast import construct_ast

print(numba.__version__)
print(numba.config.__dict__)

@njit(pipeline_class=GetSSACompiler)
def test_loop():
    x = 0
    for i in range(10):
        x += 1
    return x


ssa_by_blocks.clear()
blocks.clear()
func_ir = None

test_loop()
print("SSA Statements Grouped by Block:")
for blk_offset, ssa_list in ssa_by_blocks.items():
    print(f"Block {blk_offset}:")
    for stmt in ssa_list:
        print(f"  {stmt}")
print()

viz_ast_and_cfg(blocks, test_loop)
