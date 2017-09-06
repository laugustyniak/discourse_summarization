import unittest

from aspects.utilities.transformations import flatten_list


class UtilsTest(unittest.TestCase):
    def test_list_flattening(self):
        l = [[u'car', u'phone'], [u'car', u'phone']]
        list_expected = [u'car', u'phone', u'car', u'phone']
        list_obtained = flatten_list(l)
        self.assertEqual(list_expected, list_obtained)
