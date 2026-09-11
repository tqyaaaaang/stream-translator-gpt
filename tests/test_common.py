import io
import unittest

from stream_translator_gpt.utils.common import write_console_utf8


class ConsoleOutputTest(unittest.TestCase):

    def test_write_console_utf8_falls_back_to_underlying_binary_stream(self):
        class LegacyStream:

            encoding = 'ascii'

            def __init__(self):
                self.buffer = io.BytesIO()

            def write(self, value):
                value.encode(self.encoding)

            def flush(self):
                pass

        stream = LegacyStream()

        write_console_utf8('テスト。', stream=stream)

        self.assertEqual(stream.buffer.getvalue(), 'テスト。\n'.encode('utf-8'))


if __name__ == '__main__':
    unittest.main()
