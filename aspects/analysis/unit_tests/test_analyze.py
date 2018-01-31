import unittest

import pandas as pd

from aspects.analysis.analyze import get_count_from_series


class AnalyzeTest(unittest.TestCase):
    def test_list_flattening(self):
        df = pd.DataFrame()
        df['a'] = [[u'car', u'phone'], [u'car', u'phone']]
        counter_expected = {u'car': 2, u'phone': 2}
        counter_obtained = get_count_from_series(df.a)
        self.assertEqual(counter_expected, counter_obtained)
