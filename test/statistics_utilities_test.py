import unittest

from src.statistics_utilities import column_stat_summaries


class MyTestCase(unittest.TestCase):
    def test_column_stat_summaries(self):
        dataset = [[3.393533211, 2.331273381, 0],
                   [3.110073483, 1.781539638, 0],
                   [1.343808831, 3.368360954, 0],
                   [3.582294042, 4.67917911, 0],
                   [2.280362439, 2.866990263, 0],
                   [7.423436942, 4.696522875, 1],
                   [5.745051997, 3.533989803, 1],
                   [9.172168622, 2.511101045, 1],
                   [7.792783481, 3.424088941, 1],
                   [7.939820817, 0.791637231, 1]]

        self.assertEqual(
            column_stat_summaries(dataset),
            [(5.178333386499999, 2.7665845055177263, 10), (2.9984683241, 1.218556343617447, 10)]
        )


if __name__ == '__main__':
    unittest.main()
