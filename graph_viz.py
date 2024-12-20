import ast
import inspect
import os

import graphviz
from graphviz import Digraph
from numba.core.analysis import compute_cfg_from_blocks

from recontruct_ast import construct_ast


def get_ast(func):
    source = inspect.getsource(func)
    tree = ast.parse(source)
    print("AST Tree:", ast.dump(tree, indent=4))
    return tree


def build_ast_graph(node, graph=None, parent=None):
    if graph is None:
        graph = Digraph(comment="AST")

    # context of the node, for example, Store(), Load()
    node.ctx = None

    node_id = str(id(node))
    label = type(node).__name__

    # Set node label for specific AST types
    if isinstance(node, ast.Constant):
        label += f"\\n{node.value}"
    elif isinstance(node, ast.Name):
        label += f"\\n{node.id}"
    elif isinstance(node, ast.BinOp):
        label += f"\\n{type(node.op).__name__}"
    elif isinstance(node, ast.Attribute):
        label += f"\\n{node.attr}"

    graph.node(node_id, label)

    # Draw an edge between current node and its parent node
    if parent is not None:
        graph.edge(parent, node_id)

    # Recursively draw children of current node (draw subtrees)
    for child_name, child in ast.iter_fields(node):
        if isinstance(child, ast.AST):
            build_ast_graph(child, graph, node_id)
        elif isinstance(child, list):
            for item in child:
                if isinstance(item, ast.AST):
                    build_ast_graph(item, graph, node_id)

    return graph


def viz_ast(filename, ast_node):
    graph = build_ast_graph(ast_node)
    graph.render(filename, format='png', cleanup=True)


def viz_cfg(filename, blocks):
    """
    Args:
        blocks: Basic blocks generated by Numba (in SSA form)
        filename: Name of the CFG file.

    Returns:
        Graphviz graph representing the CFG
    """
    # This is function is provided by Numba
    cfg = compute_cfg_from_blocks(blocks)

    graph = graphviz.Digraph("ControlFlowGraph")

    # Add nodes (basic blocks)
    for block_offset, block in blocks.items():
        block_label = f"Block {block_offset}\\n"
        block_label += "\\n".join([str(stmt) for stmt in block.body])
        graph.node(str(block_offset), label=block_label, shape="box")

    # Add edges (control flow between blocks)
    for block_offset, _ in blocks.items():
        for succ, _ in cfg.successors(block_offset):
            graph.edge(str(block_offset), str(succ))

    # Render and display
    graph.render(filename, format='png', cleanup=True)
    return graph

def viz_ast_and_cfg(blocks, target_func):
    func_name = target_func.__name__
    output_dir = f"{func_name}_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    tree = get_ast(target_func)
    ast_filename = os.path.join(output_dir, f"{func_name}_ast")
    viz_ast(ast_filename, tree)
    cfg_filename = os.path.join(output_dir, f"{func_name}_cfg")
    viz_cfg(cfg_filename, blocks)
    ast_recon_filename = os.path.join(output_dir, f"{func_name}_ast_reconstructed")
    tree_recon = construct_ast(blocks)
    viz_ast(ast_recon_filename, tree_recon)
