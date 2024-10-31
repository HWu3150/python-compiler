import ast
import operator
import warnings

from numba import njit, prange
from numba.core import ir, ir_utils, config, errors
from numba.core.compiler import CompilerBase, DefaultPassBuilder
from numba.core.compiler_machinery import FunctionPass, register_pass
from numba.core.untyped_passes import IRProcessing, ReconstructSSA
from numba.core.ir_utils import *
from numbers import Number
import numba


# Register this pass with the compiler framework, declare that it will not
# mutate the control flow graph and that it is not an analysis_only pass (it
# potentially mutates the IR).
@register_pass(mutates_CFG=False, analysis_only=False)
class ConstsAddOne(FunctionPass):
    _name = "consts_add_one" # the common name for the pass

    def __init__(self):
        FunctionPass.__init__(self)

    # implement method to do the work, "state" is the internal compiler
    # state from the CompilerBase instance.
    def run_pass(self, state):
        func_ir = state.func_ir # get the FunctionIR object
        mutated = False # used to record whether this pass mutates the IR
        # walk the blocks
        for blk in func_ir.blocks.values():
            tt = blk.find_insts(ir.Branch)
            # find LHS of assignment
            pp = tt
            print(pp)
            # find the assignment nodes in the block and walk them
            for assgn in blk.find_insts(ir.Assign):
                # if an assignment value is a ir.Consts
                #print(assgn.value)
                if isinstance(assgn.value, ir.Const):
                    const_val = assgn.value
                    # if the value of the ir.Const is a Number
                    if isinstance(const_val.value, Number):
                        # then add one!
                        const_val.value += 1
                        mutated |= True
        return mutated  # return True if the IR was mutated, False if not.


def get_rhs_vars(stmt):
    var_list = []
    if isinstance(stmt, ir.Assign):
        if not isinstance(stmt.value, ir.Const):
            if isinstance(stmt.value, ir.Expr):
                # get list of vars from the Expr
                expr_vars = ir.Expr.list_vars(stmt.value)
                for var in expr_vars:
                    var_list.append(var.name)
            else:
                var_list.append(stmt.value.name)
    elif isinstance(stmt, ir.Return):
        var_list.append(stmt.value.name)
    elif isinstance(stmt, ir.Expr):
        var_list.append(ir.Expr.list_vars(stmt))
    else:
        raise ValueError("Unexpected statement type: {}".format(type(stmt)))
    return var_list


def get_lhs_vars(stmt):
    if isinstance(stmt, ir.Assign):
        return stmt.target
    elif isinstance(stmt, ir.Return):
        return stmt.value
    elif isinstance(stmt, ir.Expr):
        return stmt
    else:
        raise ValueError("Unexpected statement type: {}".format(type(stmt)))


@register_pass(mutates_CFG=False, analysis_only=False)
class PrintAssignments(FunctionPass):
    _name = "dead_code_elimination1"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        # state contains the FunctionIR to be mutated,
        mutate = True
        new_len, cur_len = 0, 0
        func_ir = state.func_ir
        vars = ir_utils.get_name_var_table(func_ir.blocks)

        for blk in func_ir.blocks.values():
            cur_len = len(blk.body)
            while cur_len != new_len:
                used_vars = []
                cur_len = len(blk.body)
                for stmt in blk.body:
                    print(stmt)
                    used_vars.append(get_rhs_vars(stmt))
                used_vars = [item for sublist in used_vars for item in sublist]

                new_body = []
                # iterate over each statement in the block
                for stmt in blk.body:
                    target_var = get_lhs_vars(stmt)
                    if target_var.name not in used_vars:
                        mutate = True  # the pass will mutate the IR
                        continue  # skip this statement
                    new_body.append(stmt)  # keep this statement
                    print(stmt)
                blk.body = new_body  # update the block with new statements
                new_len = len(blk.body)
        return mutate  # the pass has not mutated the IR


class MyCompiler(CompilerBase):  # custom compiler extends from CompilerBase

    def define_pipelines(self):
        pm = DefaultPassBuilder.define_nopython_pipeline(self.state)
        #pm.add_pass_after(ConstsAddOne, IRProcessing)
        pm.add_pass_after(PrintAssignments, ReconstructSSA)
        #pm.add_pass_after(PrintAssignments, ReconstructSSA)
        pm.finalize()
        return [pm]


@njit(pipeline_class=MyCompiler)
def dce_test():
    a = 10
    b = 20
    c = 40
    d = c + a
    e = d + b
    c = a + b
    return c


# test SpMV csr
# generate a random sparse matrix CSR format
print(numba.__version__)
c = dce_test()
print(c)



