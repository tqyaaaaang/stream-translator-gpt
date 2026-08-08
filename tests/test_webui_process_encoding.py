import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

from webui.process_utils import build_utf8_subprocess_env, popen_utf8, run_utf8


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UNICODE_LINE = '翻译：結城さくな正在唱歌。'


class WebuiSubprocessEncodingTest(unittest.TestCase):

    def test_environment_forces_utf8_and_preserves_pythonpath(self):
        with mock.patch.dict(os.environ, {
                'PYTHONUTF8': '0',
                'PYTHONIOENCODING': 'cp1252:replace',
                'PYTHONPATH': 'existing-path',
        }):
            env = build_utf8_subprocess_env(str(PROJECT_ROOT))

        self.assertEqual(env['PYTHONUTF8'], '1')
        self.assertEqual(env['PYTHONIOENCODING'], 'utf-8')
        self.assertEqual(env['PYTHONPATH'].split(os.pathsep), [str(PROJECT_ROOT), 'existing-path'])

    @unittest.skipUnless(sys.platform == 'win32', 'Exercises the Windows cp1252 redirected-pipe failure mode')
    def test_real_windows_pipe_reproduces_cp1252_replacement_then_preserves_unicode(self):
        child = [sys.executable, '-c', f'print({UNICODE_LINE!r})']
        legacy_env = os.environ.copy()
        legacy_env['PYTHONUTF8'] = '0'
        legacy_env['PYTHONIOENCODING'] = 'cp1252:replace'

        legacy = subprocess.run(child,
                                capture_output=True,
                                text=True,
                                encoding='cp1252',
                                errors='replace',
                                env=legacy_env,
                                check=True)
        self.assertNotIn(UNICODE_LINE, legacy.stdout)
        self.assertIn('?', legacy.stdout)

        fixed = run_utf8(child, capture_output=True, env=legacy_env, check=True)
        self.assertEqual(fixed.stdout.rstrip('\r\n'), UNICODE_LINE)

    def test_popen_utf8_reads_real_child_pipe_without_mojibake(self):
        child = [
            sys.executable, '-c',
            f'import sys; print(sys.stdout.encoding); print({UNICODE_LINE!r})'
        ]
        process = popen_utf8(child,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT,
                             env=build_utf8_subprocess_env(str(PROJECT_ROOT)))
        output, _ = process.communicate(timeout=10)

        self.assertEqual(process.returncode, 0)
        encoding, text = output.splitlines()
        self.assertEqual(encoding.lower().replace('_', '-'), 'utf-8')
        self.assertEqual(text, UNICODE_LINE)

    def test_invalid_child_bytes_are_replaced_instead_of_crashing_reader(self):
        child = [sys.executable, '-c', "import os; os.write(1, b'valid\\xfftail')"]
        result = run_utf8(child, capture_output=True, check=True)
        self.assertEqual(result.stdout, 'valid\ufffdtail')


if __name__ == '__main__':
    unittest.main()
