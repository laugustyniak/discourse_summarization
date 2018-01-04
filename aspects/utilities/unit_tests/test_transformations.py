import unittest

from aspects.utilities.transformations import merge_dicts_by_key


class TransformationsTest(unittest.TestCase):
    def test_merge_dicts_by_key(self):
        r = {'phone': 0.32456160227748465, 'screen': 0.17543839772251535,
             'speaker': 0.17543839772251535, 'apple': 0.32456160227748465}
        dir_moi = {'phone': 1, 'apple': 2}

        merged_dicts = merge_dicts_by_key(dir_moi, r, dir_moi)
        dict_expected = {'phone': [1, 0.32456160227748465, 1],
                         'screen': [0.17543839772251535],
                         'speaker': [0.17543839772251535],
                         'apple': [2, 0.32456160227748465, 2]
                         }
        self.assertEqual(merged_dicts, dict_expected)
