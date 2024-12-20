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


def fetch_node_def(var_name, map, ast_ctx):
    if var_name in map:
        return map[var_name]
    else:
        return ast.Name(id=var_name, ctx=ast_ctx)


def convert_stmt_to_node(stmt,
                         inplace_ops,
                         arithmetic_ops,
                         compare_ops,
                         aug_assigns_targets,
                         ssa_var_map,
                         args):
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

        # Function argument
        if isinstance(stmt.value, ir.Arg):
            args.append(ast.arg(
                arg=stmt.value.name
            ))

        if isinstance(stmt.value, ir.Global):
            if stmt.value.name == 'range':
                ssa_var_map[stmt.target.name] = ast.Name(id='range', ctx=ast.Load())
            if stmt.value.name == 'np':
                ssa_var_map[stmt.target.name] = ast.Name(id='np', ctx=ast.Load())

        if isinstance(stmt.value, ir.Expr) and stmt.value.op == 'call':
            call_body = stmt.value._kws
            func_name = call_body['func'].name
            if func_name in ssa_var_map:
                func_node = ssa_var_map[func_name]
                call_node = ast.Call(
                    func=func_node,
                    args=[],
                    keywords=[]
                )
                for arg in call_body['args']:
                    call_node.args.append(fetch_node_def(arg.name, ssa_var_map, ast.Load()))
                if isinstance(func_node, ast.Name) and func_node.id == 'range':
                    return ast.For(
                        target=None,
                        iter=call_node,
                        body=[],
                        orelse=[]
                    )
                else:
                    return ast.Assign(
                        targets=[ast.Name(id=stmt.target.name, ctx=ast.Store())],
                        value=call_node
                    )

        # R.H.S of Assign is a Variable
        if isinstance(stmt.value, ir.Var):
            if 'phi' in stmt.value.name or 'iter' in stmt.value.name:
                return None
            if stmt.target not in aug_assigns_targets and stmt.value not in aug_assigns_targets:
                return ast.Assign(
                    targets=[ast.Name(id=stmt.target.name, ctx=ast.Store())],
                    value=ast.Name(id=stmt.value.name, ctx=ast.Load())
                )

        # R.H.S of Assign is Const
        elif isinstance(stmt.value, ir.Const):
            # SSA var created for constant
            if '$const' in stmt.target.name:
                ssa_var_map[stmt.target.name] = ast.Constant(value=stmt.value.value)
            else:
                return ast.Assign(
                    targets=[ast.Name(id=stmt.target.name, ctx=ast.Store())],
                    value=ast.Constant(value=stmt.value.value)
                )

        # R.H.S of Assign is a binop
        elif isinstance(stmt.value, ir.Expr) and stmt.value.op == 'binop':
            # binop is an arithmetic operation
            if stmt.value.fn in arithmetic_ops:
                return ast.Assign(
                    targets=[ast.Name(id=stmt.target.name, ctx=ast.Store())],
                    value=ast.BinOp(
                        left=fetch_node_def(stmt.value.lhs.name, ssa_var_map, ast.Load()),
                        op=arithmetic_ops[stmt.value.fn],
                        right=fetch_node_def(stmt.value.rhs.name, ssa_var_map, ast.Load()),
                    ),
                )
            # binop is a compare operation
            if stmt.value.fn in compare_ops:
                return ast.Assign(
                    targets=[ast.Name(id=stmt.target.name, ctx=ast.Store())],
                    value=ast.Compare(
                        left=fetch_node_def(stmt.value.lhs.name, ssa_var_map, ast.Load()),
                        ops=[compare_ops[stmt.value.fn]],
                        comparators=[fetch_node_def(stmt.value.rhs.name, ssa_var_map, ast.Load())],
                    ),
                )

        # R.H.S of Assign is inplace binop, e.g., x += 1 -> $tmp = inplace(add, x, 1)
        elif isinstance(stmt.value, ir.Expr) and stmt.value.op == 'inplace_binop':
            aug_assigns_targets.append(stmt.target)
            return ast.AugAssign(
                target=fetch_node_def(stmt.value.lhs.name, ssa_var_map, ast.Store()),
                op=inplace_ops[stmt.value.fn],
                value=fetch_node_def(stmt.value.rhs.name, ssa_var_map, ast.Load()),
            )

        # Expr
        elif isinstance(stmt.value, ir.Expr) and stmt.value.op == 'cast':
            return ast.Assign(
                targets=[ast.Name(id=stmt.target.name, ctx=ast.Store())],
                value=ast.Name(id=stmt.value._kws['value'], ctx=ast.Load()),
            )

        # Build tuple, e.g., [i, j] of A[i, j]
        elif isinstance(stmt.value, ir.Expr) and stmt.value.op == 'build_tuple':
            tuple_node = ast.Tuple(
                elts=[],
                ctx=ast.Load()
            )
            for var in stmt.value.items:
                tuple_node.elts.append(fetch_node_def(var.name, ssa_var_map, ast.Load()))
            ssa_var_map[stmt.target] = tuple_node
            # return tuple_node

        # Subscripting, e.g., A[i, j]
        elif isinstance(stmt.value, ir.Expr) and stmt.value.op == "getitem":
            value_node = ssa_var_map.get(stmt.value.value)
            slice_node = ssa_var_map.get(stmt.value.index)
            const_slice_node = fetch_node_def(stmt.value.index.name, ssa_var_map, ast.Load())
            subscript_node = ast.Subscript(
                value=ast.Name(id=stmt.value.value, ctx=ast.Load()) if value_node is None else value_node,
                slice=None,
                ctx=ast.Load()
            )

            if slice_node is not None:
                subscript_node.slice = slice_node
            else:
                subscript_node.slice = const_slice_node

            return ast.Assign(
                targets=[ast.Name(id=stmt.target.name, ctx=ast.Store())],
                value=subscript_node
            )

        # Get attribute, e.g., A.shape[0]
        elif isinstance(stmt.value, ir.Expr) and stmt.value.op == "getattr":
            value_node = ssa_var_map.get(stmt.value.value)
            attribute_node = ast.Attribute(
                value=ast.Name(id=stmt.value.value, ctx=ast.Load()) if value_node is None else value_node,
                attr=stmt.value.attr,
                ctx=ast.Load()
            )
            ssa_var_map[stmt.target] = attribute_node
            # return attribute_node

    # Return
    elif isinstance(stmt, ir.Return):
        return ast.Return(
            value=ast.Name(id=stmt.value.name, ctx=ast.Load())
        )

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

def process_for_node(label, loops, cfg, blocks, for_node):
    """
    Find the loop variant variable of the AST For node.
    e.g. var i of the for loop: for i in range(10)

    Args:
        label: Label of the basic block.
        loops: A dictionary mapping loop headers to loop information.
        cfg: CFG.
        blocks: Basic blocks.
        for_node: AST For node to be processed.

    Returns:
        Processed For node.
    """
    loop_header_label = int()
    for src, _ in cfg.successors(label):
        if src in loops:
            loop_header_label = src
            break
    true_condition_label = blocks[loop_header_label].body[-1].truebr
    loop_var = blocks[true_condition_label].body[0].target

    for_node.target = ast.Name(
        id=loop_var,
        ctx=ast.Load()
    )

    return for_node

def convert_block_to_ast(label, loops, cfg, blocks, args):
    """
    Args:
        label: Label of the basic block.
        loops: A dictionary mapping loop headers to loop information.
        cfg: CFG.
        blocks: Basic blocks.
        args: Arguments of the function.

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

    block = blocks[label]

    ast_nodes = list()
    unnecessary_vars = compute_unnecessary_variables(block, label, loops, cfg)
    aug_assign_targets = list()
    ssa_var_map = defaultdict()

    # Convert regular stmt to AST node
    for stmt in block.body:
        if isinstance(stmt, ir.Assign) and stmt.target in unnecessary_vars:
            continue
        if isinstance(stmt, ir.Branch) and stmt.cond in unnecessary_vars:
            continue
        ast_node = convert_stmt_to_node(stmt,
                                        inplace_ops,
                                        arithmetic_ops,
                                        compare_ops,
                                        aug_assign_targets,
                                        ssa_var_map,
                                        args)
        if ast_node is not None:
            if isinstance(ast_node, ast.For):
                for_node = process_for_node(label, loops, cfg, blocks, ast_node)
                ast_nodes.append(for_node)
            else:
                ast_nodes.append(ast_node)

    # Deal with Branch statement
    # Branch could be in loop entry or body
    # Branch must be in the last line in a block
    if isinstance(block.body[-1], ir.Branch):
        # If the block is loop entry
        if is_loop_entry(label, loops):
            cond_stmt_node = ast_nodes.pop()
            ast_nodes.append(insert_while(cond_stmt_node))
        # elif is_if_branch(label, loops, cfg):
        #     cond_stmt_node = ast_nodes.pop()
        #     ast_nodes.append(insert_if(cond_stmt_node))

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
    loop_header_to_body = cfg._find_loops()
    arg_list = list()

    # Compute AST node list of each block
    all_ast_nodes = defaultdict(list)
    for label, block in blocks.items():
        ast_nodes = convert_block_to_ast(label, loop_header_to_body, cfg, blocks, arg_list)
        all_ast_nodes[label] = ast_nodes

    loop_entry_to_body = defaultdict(list)
    loop_entry_to_header = defaultdict()
    for header, loop in loop_header_to_body.items():
        for entry in loop.entries:
            loop_entry_to_body[entry] = loop.body
            loop_entry_to_header[entry] = header

    # Compute dominator tree
    dom_tree = cfg.dominator_tree()

    def recursive_construct(block_label, depth=0):
        """
        Recursively construct AST.

        Args:
            block_label: Label of the basic block.
            depth: AST depth.

        Returns:
            AST node list.
        """
        nodes = all_ast_nodes[block_label]  # List of AST nodes of current block
        child_labels = dom_tree.get(block_label, set())  # Fetch dominated children blocks

        # If current block is loop entry
        if block_label in loop_entry_to_body:
            loop_node = nodes[-1]   # Last node must be loop node
            if isinstance(loop_node, ast.For):
                for_header_label = list(dom_tree[block_label])[0]
                dom_tree[block_label] = dom_tree[for_header_label]
                del dom_tree[for_header_label]
                child_labels = dom_tree.get(block_label, set())

            # Process blocks in loop body
            for child_label in child_labels:
                # If child block belongs to loop body
                if child_label in loop_entry_to_body[block_label]:
                    loop_node.body.extend(recursive_construct(child_label, depth + 1))
                else:
                    # Skip non-loop-body blocks for now
                    pass

        # Process same-depth dominated children blocks
        for child_label in child_labels:
            if child_label not in loop_entry_to_body.get(block_label, set()):  # not in loop body
                nodes.extend(recursive_construct(child_label, depth))

        return nodes

    # Function body
    func_body = recursive_construct(0)

    # Function arguments
    args = ast.arguments(
        posonlyargs=[],
        args=arg_list,
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[]
    )

    # Function node
    func_def = ast.FunctionDef(
        name="func",
        args=args,
        body=func_body,
        decorator_list=[],
        returns=None
    )

    # Root node
    module = ast.Module(
        body=[func_def],
        type_ignores=[]
    )

    return module