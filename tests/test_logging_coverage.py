import ast
import pathlib
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]


class _PrintCallVisitor(ast.NodeVisitor):

    def __init__(self):
        self.function_stack = []
        self.calls = []

    def visit_FunctionDef(self, node):
        self.function_stack.append(node.name)
        self.generic_visit(node)
        self.function_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == 'print':
            function_name = self.function_stack[-1] if self.function_stack else None
            self.calls.append((function_name, node.lineno))
        self.generic_visit(node)


class LoggingCoverageTest(unittest.TestCase):

    def test_project_print_calls_are_only_business_output(self):
        allowed_functions = {
            pathlib.Path('stream_translator_gpt/audio_transcriber.py'): {'loop'},
            pathlib.Path('stream_translator_gpt/main.py'): {'_print_audio_devices'},
            pathlib.Path('stream_translator_gpt/result_exporter.py'): {'loop'},
        }
        unexpected = []

        source_roots = [PROJECT_ROOT / 'stream_translator_gpt', PROJECT_ROOT / 'webui']
        for source_root in source_roots:
            for path in source_root.rglob('*.py'):
                relative_path = path.relative_to(PROJECT_ROOT)
                if 'simul_streaming' in relative_path.parts:
                    continue
                visitor = _PrintCallVisitor()
                visitor.visit(ast.parse(path.read_text(encoding='utf-8'), filename=str(relative_path)))
                allowed = allowed_functions.get(relative_path, set())
                unexpected.extend(
                    f'{relative_path}:{line}' for function_name, line in visitor.calls if function_name not in allowed)

        self.assertEqual(unexpected, [])


if __name__ == '__main__':
    unittest.main()
