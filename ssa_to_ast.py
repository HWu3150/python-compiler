# Mapping from Numba SSA operation to Python AST node
import ast
import inspect
import re

from numba.core import ir


"""
Mapping between SSA operators to AST operators
"""
op_map = {
    'add': ast.Add,
    'sub': ast.Sub,
    'mul': ast.Mult,
    'div': ast.Div
}


def ssa_to_ast(ssa_statements, func):
    """
    Convert Numba SSA to Python AST.

    Args:
        ssa_statements: SSA statements of a function.
        func: A function from whose SSA we want to reconstruct its Python AST.

    Returns:
        Function's reconstructed Python AST

    """
    expr_dict = {}              # stores rhs expressions
    const_dict = {}             # stores intermediate variables used to store constants
    assign_targets = set()      # prevent substituting variables by its assignment statement
    var_map = {}                # mapping between original variable and its SSAs
    func_body = []

    # get function name and args
    func_name = func.__name__
    args = [arg.name for arg in inspect.signature(func).parameters.values()]

    # In this demo, I only considered the cases that an expression is either
    # a variable assignment or a return statement
    for stmt in ssa_statements:
        if isinstance(stmt, ir.Assign):
            target_name = stmt.target.name

            # store intermediate constant variables created by SSA
            if target_name.startswith('$'):
                const_dict[target_name] = stmt.value.value
                continue

            # store mapping between SSA version names and base names
            base_name = re.sub(r'\.\d+$', '', target_name)
            var_map[target_name] = base_name

            # store rhs of the expression
            expr_dict[target_name] = stmt.value

            # create an Assign AST node
            target_node = ast.Name(id=var_map.get(target_name, target_name), ctx=ast.Store())
            value_node = build_ast(target_name, expr_dict, const_dict, var_map, assign_targets)
            assign_node = ast.Assign(targets=[target_node], value=value_node)
            func_body.append(assign_node)

            assign_targets.add(target_name)

        elif isinstance(stmt, ir.Return):
            return_node = None
            if stmt.value.name in const_dict:                # returning a variable
                tmp_name = const_dict[stmt.value.name].name  # if returning a intermediate constant variable

                # find its original form, then check if the original form is an SSA
                if tmp_name in var_map:
                    return_var_node = ast.Name(id=var_map[tmp_name], ctx=ast.Load())
                    return_node = ast.Return(value=return_var_node)
                else:
                    return_var_node = ast.Name(id=tmp_name, ctx=ast.Load())
                    return_node = ast.Return(value=return_var_node)

            # returning an expression
            else:
                return_expr = stmt.value.name
                return_node = ast.Return(value=build_ast(return_expr, expr_dict, const_dict, var_map, assign_targets))

            func_body.append(return_node)

    # Add function parameters
    arg_nodes = [ast.arg(arg=arg, annotation=None) for arg in args]
    args_node = ast.arguments(
        posonlyargs=[], args=arg_nodes, vararg=None, kwonlyargs=[],
        kw_defaults=[], kwarg=None, defaults=[]
    )

    # Build function definition node
    func_def_node = ast.FunctionDef(
        name=func_name,
        args=args_node,
        body=func_body,
        decorator_list=[]
    )

    ssa_statements.clear()

    return ast.Module(body=[func_def_node], type_ignores=[])


def build_ast(target_name, expr_dict, const_dict, var_map, assign_targets):
    """
    Recursively reconstruct AST

    Args:
        target_name: LHS side of an expression.
        expr_dict:
        const_dict:
        var_map:
        assign_targets:

    Returns:
        AST node of the expression.

    """
    if target_name in const_dict and target_name.startswith('$'):
        return ast.Constant(value=const_dict[target_name])

    expr = expr_dict[target_name]
    # print('new run')
    # print('assign targets:', assign_targets)
    # print(target_name, const_dict)
    # print('expression:', expr)

    if isinstance(expr, ir.Const):
        if target_name in assign_targets:
            return ast.Name(id=target_name, ctx=ast.Load())
        return ast.Constant(value=expr.value)

    elif isinstance(expr, ir.Var):
        return ast.Name(id=var_map[expr.Name], ctx=ast.Load())

    # if the expression contains another expression, recursively build the AST
    elif isinstance(expr, ir.Expr):
        if target_name in assign_targets:
            return ast.Name(id=target_name, ctx=ast.Load())

        if expr.op == 'cast':
            return None

        left_node = build_ast(var_map.get(expr.lhs.name, expr.lhs.name), expr_dict, const_dict, var_map, assign_targets)
        right_node = build_ast(var_map.get(expr.rhs.name, expr.rhs.name), expr_dict, const_dict, var_map,
                               assign_targets)
        op_node = op_map[expr.fn.__name__]()  # binary ops are encapsulated as Python fn
        return ast.BinOp(left=left_node, op=op_node, right=right_node)

    else:
        raise ValueError(f"Unsupported expression type: {type(expr)}")
