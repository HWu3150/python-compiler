from numba.core.compiler import CompilerBase, DefaultPassBuilder
from numba.core.compiler_machinery import register_pass, FunctionPass
from numba.core.untyped_passes import IRProcessing, ReconstructSSA

ssa_statements = []  # stores SSA statements of a function


@register_pass(mutates_CFG=False, analysis_only=True)
class GetSSAPass(FunctionPass):
    _name = "get_ssa"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        """
        A compiler pass that retrieves SSA statements of a given function.

        Args:
            state: Current state of the compiler.

        Returns:
            False, indicating code structure is not modified.

        """
        for blk in state.func_ir.blocks.values():
            for stmt in blk.body:
                ssa_statements.append(stmt)
        return False


class GetSSACompiler(CompilerBase):

    def define_pipelines(self):
        """
        Get Numba SSA statements of the function after IRProcessing (modified).

        Returns:
            Pipeline manager.

        """
        pm = DefaultPassBuilder.define_nopython_pipeline(self.state)
        # pm.add_pass_after(GetSSAPass, ReconstructSSA)
        pm.add_pass_after(GetSSAPass, IRProcessing)
        pm.finalize()
        return [pm]
