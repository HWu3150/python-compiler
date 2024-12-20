from numba import njit

from get_ssa_compiler import GetSSACompiler, ssa_by_blocks, blocks
from graph_viz import viz_ast_and_cfg

import numba

print(numba.__version__)
print(numba.config.__dict__)

@njit(pipeline_class=GetSSACompiler)
def test_while_loop():
    x = 0
    while x < 5:
        y = 1
        x += y
        while x % 2 == 0:
            x += 1
        x += 1
    return x


ssa_by_blocks.clear()
blocks.clear()
func_ir = None

test_while_loop()
print("SSA Statements Grouped by Block:")
for blk_offset, ssa_list in ssa_by_blocks.items():
    print(f"Block {blk_offset}:")
    for stmt in ssa_list:
        print(f"  {stmt}")
print()

viz_ast_and_cfg(blocks, test_while_loop)
