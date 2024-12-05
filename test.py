from numba import njit

from get_ssa_compiler import GetSSACompiler, ssa_by_blocks, blocks
from graph_viz import viz_cfg, viz_ast, get_ast

import numba

from recontruct_ast import construct_ast

print(numba.__version__)
print(numba.config.__dict__)

@njit(pipeline_class=GetSSACompiler)
def test_loop():
    x = 0
    while x < 5:
        while x % 2 == 0:
            x += 1
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

# Visualize AST of the source code
viz_ast('original AST', get_ast(test_loop))
# Visualize CFG of the SSA code
viz_cfg(blocks)

# Convert to AST
astree = construct_ast(blocks)
viz_ast('Reconstructed AST', astree)
