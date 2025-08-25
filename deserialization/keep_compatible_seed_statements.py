"""
Description:
    Given a custom (i.e., user-written or LLM-written) test file, the PynguinSeedCleaner
    will remove statements of the test file that are not compatible with Pynguin's internal
    test representation (e.g., branching statements). See paper for details.

Author:
    Konstantinos Kitsios <konstantinos.kitsios@uzh.ch>

Affiliation:
    University of Zurich

Date:
    22 Aug. 2025
"""

import ast
import astor
import sys
from pathlib import Path

# Allowed RHS types for assignment
ALLOWED_ASSIGN_TYPES = (ast.Call, ast.Constant, ast.List, ast.Tuple, ast.Dict, ast.Set, ast.UnaryOp)

# Disallowed statements entirely
DISALLOWED_BLOCK_TYPES = (
    ast.For, ast.While, ast.If, ast.With, ast.Try, ast.Return, ast.Raise, ast.Lambda,
)


class PynguinSeedCleaner(ast.NodeTransformer):
    def __init__(self, module_under_test: str):
        self.module_under_test = module_under_test
        self.defined_vars = set()
        self.used_aliases = set()
        self.collected_imports = []
        self.alias_to_module = {}

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.defined_vars.clear()
        self.used_aliases.clear()
        new_body = []

        for stmt in node.body:
            if isinstance(stmt, DISALLOWED_BLOCK_TYPES):
                continue

            if isinstance(stmt, ast.Assign):
                if not isinstance(stmt.value, ALLOWED_ASSIGN_TYPES):
                    continue

                if isinstance(stmt.value, ast.Call) and not self._call_allowed(stmt.value):
                    continue

                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        self.defined_vars.add(target.id)

                self._track_aliases(stmt.value)
                new_body.append(stmt)

            elif isinstance(stmt, ast.Assert):
                vars_in_assert = {n.id for n in ast.walk(stmt.test) if isinstance(n, ast.Name)}
                if vars_in_assert.issubset(self.defined_vars):
                    self._track_aliases(stmt.test)
                    new_body.append(stmt)

            elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                if self._call_allowed(stmt.value):
                    self._track_aliases(stmt.value)
                    new_body.append(stmt)

            elif isinstance(stmt, ast.Import):
                self._record_import_aliases(stmt)
                self.collected_imports.append(stmt)

        if not new_body:
            return None  # Drop function entirely if nothing is valid

        node.body = self._inject_valid_imports() + new_body
        return node

    def _call_allowed(self, call_node):
        """Returns True if the function call is allowed (known or in-module)."""
        if isinstance(call_node.func, ast.Attribute):
            alias = getattr(call_node.func.value, 'id', None)
            if alias:
                # Allow if alias matches known import pointing to module under test
                module_path = self.alias_to_module.get(alias, "")
                return self.module_under_test in module_path
        return True  # Plain calls like f(x)

    def _track_aliases(self, expr):
        """Track all used aliases from function calls like mm.func()."""
        for n in ast.walk(expr):
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name):
                self.used_aliases.add(n.value.id)

    def _record_import_aliases(self, import_stmt: ast.Import):
        """Track alias → module path mapping."""
        for alias in import_stmt.names:
            if alias.asname:
                self.alias_to_module[alias.asname] = alias.name
            else:
                self.alias_to_module[alias.name] = alias.name

    def _inject_valid_imports(self):
        """Return only necessary imports based on usage and module-under-test."""
        valid_imports = []
        for stmt in self.collected_imports:
            keep_aliases = []
            for alias in stmt.names:
                alias_name = alias.asname or alias.name
                full_import_path = self.alias_to_module.get(alias_name, "")
                if alias_name in self.used_aliases or self.module_under_test in full_import_path:
                    keep_aliases.append(alias)
            if keep_aliases:
                stmt.names = keep_aliases
                valid_imports.append(ast.Import(names=keep_aliases))
        return valid_imports


def clean_test_file(input_path: Path, output_path: Path, module_under_test: str):
    try:
        tree = ast.parse(input_path.read_text())
    except SyntaxError as e:
        print(f"Syntax error in input file: {e}")
        sys.exit(1)

    cleaner = PynguinSeedCleaner(module_under_test)
    new_tree = cleaner.visit(tree)
    ast.fix_missing_locations(new_tree)

    code = astor.to_source(new_tree)
    output_path.write_text(code)
    print(f"Cleaned test saved to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python pynguin_seed_cleaner.py <input_test.py> <output_clean.py> <module_under_test>")
        sys.exit(1)

    input_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2])
    module_name = sys.argv[3]

    clean_test_file(input_file, output_file, module_name)
