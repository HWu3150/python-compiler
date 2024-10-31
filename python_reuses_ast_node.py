import ast

"""
Below code shows that Python internally reuses AST nodes for optimization.
Nodes like op=Add() will be reused since it does not contain extra information.
This explains why multiple BinOp nodes are directed to a same, for instance, Add node in the visualized AST.
"""

# code from dce_test()

code = """
a = 10
b = 20
c = 40
d = c + a
e = d + b
c = a + b
return c
"""

tree = ast.parse(code)

# Collect ids of all operator nodes
operator_ids = []
for node in ast.walk(tree):
    if isinstance(node, ast.BinOp):
        operator_ids.append((node.op, id(node.op)))

# Print operator type and id
for op, op_id in operator_ids:
    print(f"Operator: {type(op).__name__}, ID: {op_id}")

# Check if any IDs are duplicated
unique_ids = {op_id for _, op_id in operator_ids}
if len(unique_ids) < len(operator_ids):
    print("AST is reusing operator nodes.")
else:
    print("AST is not reusing operator nodes.")
