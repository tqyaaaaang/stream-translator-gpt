import datetime
import re
import unittest

from stream_translator_gpt import __version__


class VersionTest(unittest.TestCase):

    def test_version_is_a_stable_calendar_version(self):
        match = re.fullmatch(r'(\d{4})\.(\d{1,2})\.(\d{1,2})', __version__)
        self.assertIsNotNone(match)
        year, month, day = map(int, match.groups())
        datetime.date(year, month, day)
        self.assertNotIn('dev', __version__.lower())


if __name__ == '__main__':
    unittest.main()
