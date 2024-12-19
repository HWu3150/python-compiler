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

def is_if_branch(label, loops, cfg):
    if is_loop_entry(label, loops):
        return False
    if is_in_any_loop(label, loops):
        # find in which loop the block lies
        loop_exit = None
        for loop_label, loop in loops.items():
            if label in loop.entries or label in loop.body:
                loop_exit = list(loop.exits)[0]
                break
        # if the branch statement locates at the end
        # of the loop, then it's not an if statement
        for src, _ in cfg.predecessors(loop_exit):
            if src == label:
                return False
    return True

def compute_unnecessary_variables(block, label, loops, cfg):
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

    if is_if_branch(label, loops, cfg):
        return unnecessary_vars

    if isinstance(block.body[-1], ir.Jump):
        return unnecessary_vars

    unnecessary_vars.add(block.body[-1].cond)
    for i in range(-2, -len(block.body) - 1, -1):
        stmt = block.body[i]
        if stmt.target in unnecessary_vars:
            var_list = stmt.list_vars()
            for var in var_list:
                unnecessary_vars.add(var)
    return unnecessary_vars


def convert_stmt_to_node(stmt, aug_assigns_targets, inplace_ops, arithmetic_ops, compare_ops):
    """
    Args:
    SSA statement

    Returns:
    AST node representing the statement
    """

    # Assign
    if isinstance(stmt, ir.Assign):

        if 'phi' in stmt.target.name or 'iter' in stmt.target.name:
            return None

        if isinstance(stmt.value, ir.Global):
            if stmt.value.name == 'range':
                return ast.Call(
                    func=ast.Name(id='range', ctx=ast.Load()),
                    args=[],
                    keywords=[]
                )

        if isinstance(stmt.value, ir.Var):
            if 'phi' in stmt.value.name or 'iter' in stmt.value.name:
                return None
            if stmt.target not in aug_assigns_targets and stmt.value not in aug_assigns_targets:
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
                        ops=[compare_ops[stmt.value.fn]],
                        comparators=[ast.Name(id=stmt.value.rhs.name, ctx=ast.Load())],
                    ),
                )

        # R.H.S of Assign is inplace binop, e.g., x += 1 -> $tmp = inplace(add, x, 1)
        elif isinstance(stmt.value, ir.Expr) and stmt.value.op == 'inplace_binop':
            aug_assigns_targets.append(stmt.target)
            return ast.AugAssign(
                target=ast.Name(id=stmt.value.lhs.name, ctx=ast.Store()),
                op=inplace_ops[stmt.value.fn],
                value=ast.Name(id=stmt.value.rhs.name, ctx=ast.Load()),
            )

        # Expr
        elif isinstance(stmt.value, ir.Expr) and stmt.value.op == 'cast':
            return ast.Assign(
                targets=[ast.Name(id=stmt.target.name, ctx=ast.Store())],
                value=ast.Name(id=stmt.value._kws['value'], ctx=ast.Load()),
            )

        # Build tuple, e.g., A[i, j]
        elif isinstance(stmt.value, ir.Expr) and stmt.value.op == 'build_tuple':
            tuple_node = ast.Tuple(
                elts=[],
                ctx=ast.Load()
            )
            for var in stmt.value.items:
                tuple_node.elts.append(ast.Name(id=var.name, ctx=ast.Load()))
            return tuple_node

        elif isinstance(stmt.value, ir.Expr) and stmt.value.op == "getitem":
            return ast.Subscript(
                value=ast.Name(id=stmt.value.value, ctx=ast.Load()),
                slice=None,
                ctx=ast.Load()
            )

    # TODO, jump statement from if-else branch

    # Return
    elif isinstance(stmt, ir.Return):
        return ast.Return(
            value=ast.Name(id=stmt.value.name, ctx=ast.Load())
        )

    # Unsupported statement type
    return None


def insert_while(cond_stmt_node):
    """
    Args:
        Condition statement node

    Returns:
        AST node representing the branch statement
    """
    return ast.While(
        test=cond_stmt_node.value,  # Right subtree of the condition statement node
        body=[],  # Loop body is the node converted from the body block
        orelse=[]
    )

def insert_if(cond_stmt_node):
    return ast.If(
        test=cond_stmt_node.value,
        body=[],
        orelse=[]
    )

def insert_for(label, loops, cfg, blocks, call_node):
    loop_header_label = int()
    for src, _ in cfg.successors(label):
        if src in loops:
            loop_header_label = src
            break
    true_condition_label = blocks[loop_header_label].body[-1].truebr
    loop_var = blocks[true_condition_label].body[0].target

    return ast.For(
        target=ast.Name(id=loop_var, ctx=ast.Store()),
        iter=call_node,
        body=[],
        orelse=[]
    )

def convert_block_to_ast(block, label, loops, cfg, blocks):
    """
    Args:
        block: Basic block
        label: Label of the block
        loops: A dictionary mapping loop headers to loop information.
        cfg: CFG
        blocks: Basic blocks

    Returns:
        List of AST nodes representing the block
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

    ast_nodes = list()
    unnecessary_vars = compute_unnecessary_variables(block, label, loops, cfg)
    aug_assign_targets = list()

    # Convert regular stmt to AST node
    stmts = block.body
    i = 0
    while i < len(stmts):
        if isinstance(stmts[i], ir.Assign) and stmts[i].target in unnecessary_vars:
            i += 1
            continue
        if isinstance(stmts[i], ir.Branch) and stmts[i].cond in unnecessary_vars:
            i += 1
            continue
        ast_node = convert_stmt_to_node(stmts[i], aug_assign_targets, inplace_ops, arithmetic_ops, compare_ops)
        if ast_node is not None:
            if isinstance(ast_node, ast.Call) and isinstance(ast_node.func, ast.Name) and ast_node.func.id == 'range':
                ast_node.args.append(ast.Constant(value=stmts[i+1].value.value))
                for_node = insert_for(label, loops, cfg, blocks, ast_node)
                ast_nodes.append(for_node)
                return ast_nodes
            else:
                ast_nodes.append(ast_node)
        i += 1

    # Deal with Branch statement
    # Branch could be in loop entry or body
    # Branch must be in the last line in a block
    if isinstance(block.body[-1], ir.Branch):
        # If the block is loop entry
        if is_loop_entry(label, loops):
            cond_stmt_node = ast_nodes.pop()
            ast_nodes.append(insert_while(cond_stmt_node))
        elif is_if_branch(label, loops, cfg):
            cond_stmt_node = ast_nodes.pop()
            ast_nodes.append(insert_if(cond_stmt_node))

    return ast_nodes


def insert_temporary_node(label, ast_nodes):
    """
    For testing only. Insert a temporary parent node for a list
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

    # Compute AST node list of each block
    all_ast_nodes = defaultdict(list)
    for label, block in blocks.items():
        ast_nodes = convert_block_to_ast(block, label, loops, cfg, blocks)
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
