import ast
import operator
from collections import defaultdict

from numba.core import ir
from numba.core.analysis import compute_cfg_from_blocks


def is_loop_entry(label, loops):
    """
    Check if a block(represented by its label) is the loop entry.

    Args:
        label: Label of the block.
        loops: A dictionary mapping loop headers to loop information.

    Returns:
        True if the block is the loop entry, false otherwise.
    """
    for _, loop in loops.items():
        if label in loop.entries:
            return True
    return False


def is_in_any_loop(label, loops):
    """
    Check if a block(represented by its label) is part of any loop.

    Args:
        label: Label of the block.
        loops: A dictionary mapping loop headers to loop information.

    Returns:
        True if the block is part of any loop, false otherwise.
    """
    for _, loop in loops.items():
        if label in loop.entries or label in loop.body:
            return True
    return False


def compute_unnecessary_variables(block, label, loops):
    """
    Fold unnecessary statements in the given blocks.

    Args:
        block: Basic block.
        label: Label of the block.
        loops: A dictionary mapping loop headers to loop information.

    Returns:
        Set of unnecessary variables
    """
    unnecessary_vars = set()

    if is_loop_entry(label, loops):
        return unnecessary_vars

    if not is_in_any_loop(label, loops):
        return unnecessary_vars

    unnecessary_vars.add(block.body[-1].cond)
    for i in range(-2, -len(block.body) - 1, -1):
        stmt = block.body[i]
        if stmt.target in unnecessary_vars:
            var_list = stmt.list_vars()
            for var in var_list:
                unnecessary_vars.add(var)
    return unnecessary_vars


def convert_stmt_to_node(stmt):
    """
    Args:
    SSA statement

    Returns:
    AST node representing the statement
    """
    inplace_ops = {
        operator.iadd: ast.Add(),
        operator.isub: ast.Sub(),
        operator.imul: ast.Mult(),
        operator.itruediv: ast.Div(),
        operator.imod: ast.Mod()
    }

    arithmetic_ops = {
        operator.add: ast.Add(),
        operator.sub: ast.Sub(),
        operator.mul: ast.Mult(),
        operator.truediv: ast.Div(),
        operator.floordiv: ast.FloorDiv(),
        operator.mod: ast.Mod()
    }

    compare_ops = {
        operator.lt: ast.Lt(),
        operator.le: ast.LtE(),
        operator.eq: ast.Eq(),
        operator.ne: ast.NotEq(),
        operator.ge: ast.GtE(),
        operator.gt: ast.Gt()
    }

    # Assign
    if isinstance(stmt, ir.Assign):
        # Ignore Global

        if isinstance(stmt.value, ir.Var):
            return ast.Assign(
                targets=[ast.Name(id=stmt.target.name, ctx=ast.Store())],
                value=ast.Name(id=stmt.value.name, ctx=ast.Load()),
            )

        # R.H.S of Assign is Const
        elif isinstance(stmt.value, ir.Const):
            return ast.Assign(
                targets=[ast.Name(id=stmt.target.name, ctx=ast.Store())],
                value=ast.Constant(value=stmt.value.value),
            )

        # R.H.S of Assign is a binop
        elif isinstance(stmt.value, ir.Expr) and stmt.value.op == 'binop':
            # binop is an arithmetic operation
            if stmt.value.fn in arithmetic_ops:
                return ast.Assign(
                    targets=[ast.Name(id=stmt.target.name, ctx=ast.Store())],
                    value=ast.BinOp(
                        left=ast.Name(id=stmt.value.lhs.name, ctx=ast.Load()),
                        op=arithmetic_ops[stmt.value.fn],
                        right=ast.Name(id=stmt.value.rhs.name, ctx=ast.Load()),
                    ),
                )
            # binop is a compare operation
            if stmt.value.fn in compare_ops:
                return ast.Assign(
                    targets=[ast.Name(id=stmt.target.name, ctx=ast.Store())],
                    value=ast.Compare(
                        left=ast.Name(id=stmt.value.lhs.name, ctx=ast.Load()),
                        ops=compare_ops[stmt.value.fn],
                        comparators=ast.Name(id=stmt.value.rhs.name, ctx=ast.Load()),
                    ),
                )

        # R.H.S of Assign is inplace binop, e.g., x += 1 -> $tmp = inplace(add, x, 1)
        elif isinstance(stmt.value, ir.Expr) and stmt.value.op == 'inplace_binop':
            aug_assign = ast.AugAssign(
                target=ast.Name(id=stmt.value.lhs.name, ctx=ast.Store()),
                op=inplace_ops[stmt.value.fn],
                value=ast.Name(id=stmt.value.rhs.name, ctx=ast.Load()),
            )
            return ast.Assign(
                targets=[ast.Name(id=stmt.target.name, ctx=ast.Store())],
                value=aug_assign,
            )

        # Expr
        elif isinstance(stmt.value, ir.Expr) and stmt.value.op == 'cast':
            return ast.Assign(
                targets=[ast.Name(id=stmt.target.name, ctx=ast.Store())],
                value=ast.Name(id=stmt.value._kws['value'], ctx=ast.Load()),
            )

    # TODO, jump statement from if-else branch

    # Return
    elif isinstance(stmt, ir.Return):
        return ast.Return(
            value=ast.Name(id=stmt.value.name, ctx=ast.Load())
        )

    # Unsupported statement type
    return None


def convert_branch_to_ast(branch, cond_stmt_node):
    """
    Args:
        SSA branch statement
        Condition statement node

    Returns:
        AST node representing the branch statement
    """
    while_node = ast.While(
        test=cond_stmt_node.value,  # Right subtree of the condition statement node
        body=[],  # Loop body is the node converted from the body block
        orelse=[]
    )

    return while_node


def convert_block_to_ast(block, label, loops):
    """
    Args:
        block: Basic block
        label: Label of the block
        loops: A dictionary mapping loop headers to loop information.

    Returns:
        List of AST nodes representing the block
    """
    ast_nodes = list()

    # Convert regular stmt into AST node
    unnecessary_vars = compute_unnecessary_variables(block, label, loops)
    for stmt in block.body:
        if isinstance(stmt, ir.Assign) and stmt.target in unnecessary_vars:
            continue
        if isinstance(stmt, ir.Branch) and stmt.cond in unnecessary_vars:
            continue
        ast_node = convert_stmt_to_node(stmt)
        if ast_node is not None:
            ast_nodes.append(ast_node)

    # Deal with Branch statement
    # Branch could be in loop entry or body
    # Branch must be in the last line in a block
    if isinstance(block.body[-1], ir.Branch):
        # If the block is loop entry
        for _, loop in loops.items():
            if label in loop.entries:
                cond_stmt_node = ast_nodes.pop()
                ast_nodes.append(convert_branch_to_ast(block.body[-1], cond_stmt_node))

    return ast_nodes


def insert_temporary_node(label, ast_nodes):
    """
    For testing only. Insert a parent node for a list
    of AST nodes.

    Args:
        ast_nodes: List of AST nodes of the block.
        label: Label of the block.

    Returns:
        AST node
    """
    temp_node = ast.FunctionDef(
        name="Block {}".format(label),
        args=None,
        body=ast_nodes,
        decorator_list=[]
    )

    return temp_node


def construct_ast(blocks):
    """
    Args:
        blocks: Basic blocks.

    Returns:
        AST node representing the program
    """
    cfg = compute_cfg_from_blocks(blocks)
    loops = cfg._find_loops()

    # Use LiveIn set to find function arguments
    # live_in, _, _, _ = live_analysis(blocks)

    # Compute AST node list of each block
    all_ast_nodes = defaultdict(list)
    for label, block in blocks.items():
        ast_nodes = convert_block_to_ast(block, label, loops)
        all_ast_nodes[label] = ast_nodes

    loop_body_of_entries = defaultdict(list)
    for _, loop in loops.items():
        for entry in loop.entries:
            loop_body_of_entries[entry] = loop.body

    # Compute dominator tree
    dom_tree = cfg.dominator_tree()

    for entry_label, body in sorted(loop_body_of_entries.items(), key=lambda x: x[0], reverse=True):
        dominated_blocks = dom_tree[entry_label]
        for label in dominated_blocks:
            if label in loop_body_of_entries[entry_label]:
                for node in all_ast_nodes[entry_label]:
                    if isinstance(node, ast.While):
                        node.body.extend(all_ast_nodes[label])
            else:
                all_ast_nodes[entry_label].extend(all_ast_nodes[label])

    # Function arguments
    args = ast.arguments(
        posonlyargs=[],
        args=[],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[]
    )

    # Function body
    body = all_ast_nodes[0]

    # Function node
    func_def = ast.FunctionDef(
        name="func",
        args=args,
        body=body,
        decorator_list=[],
        returns=None
    )

    # Root node
    module = ast.Module(
        body=[func_def],
        type_ignores=[]
    )

    return module
