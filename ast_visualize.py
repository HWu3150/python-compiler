import ast
import inspect
import os

from graphviz import Digraph

from ssa_to_ast import ssa_to_ast


def get_ast(func, print_tree=False):
    """
    Get Python AST of the given function.

    Args:
        func: A function from which we want the AST.
        print_tree: A flag indicating whether to print the AST in console.

    Returns:
        Python AST of the function.

    """
    source = inspect.getsource(func)
    tree = ast.parse(source)
    if print_tree:
        print("AST Tree:", ast.dump(tree, indent=4))
    return tree


def gen_ast_visualization(node, graph=None, parent=None):
    """
    Plot a Python AST with Graphviz

    Args:
        node: Current node being visited.
        graph: A graph(canvas) for the AST.
        parent: Parent of current node.

    Returns:
        A graphviz digraph of the AST.

    """
    if graph is None:
        graph = Digraph(comment="AST")

    # context of the node, for example, Store(), Load()
    node.ctx = None

    node_id = str(id(node))
    label = type(node).__name__

    # set node name
    if isinstance(node, ast.Constant):
        label += f"\\n{node.value}"
    elif isinstance(node, ast.Name):
        label += f"\\n{node.id}"
    elif isinstance(node, ast.BinOp):
        label += f"\\n{type(node.op).__name__}"

    graph.node(node_id, label)

    # draw an edge between current node and its parent node
    if parent is not None:
        graph.edge(parent, node_id)

    # recursively draw children of current node (draw subtrees)
    for children_name, children in ast.iter_fields(node):
        if isinstance(children, ast.AST):
            gen_ast_visualization(children, graph, node_id)
        elif isinstance(children, list):
            for child in children:
                if isinstance(child, ast.AST):
                    gen_ast_visualization(child, graph, node_id)

    return graph


def visualize(func, ssa_statements=None):
    """
    Visualize the AST of the function, AST graph will be stored under results directory.

    Args:
        func: A function from which we want to visualize the AST

    Returns:

    """
    # Generate Python AST graph
    filename = f"{func.__name__} Python AST"
    output_path = os.path.join("results", filename)
    tree = get_ast(func)
    tree_viz = gen_ast_visualization(tree)
    tree_viz.render(filename=output_path, format="png", cleanup=True)

    # Generate graph of AST form SSA
    if ssa_statements is not None:
        tree_from_ssa = ssa_to_ast(ssa_statements, func)
        tree_from_ssa_viz = gen_ast_visualization(tree_from_ssa)
        filename = f"{func.__name__} AST from SSA"
        output_path = os.path.join("results", filename)
        tree_from_ssa_viz.render(filename=output_path, format="png", cleanup=True)

