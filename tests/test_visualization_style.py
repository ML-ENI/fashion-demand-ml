import sys
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.visualization.style import es, format_number, format_percent, localize_axis_numbers


class SpanishNumberFormatTests(unittest.TestCase):
    def test_number_and_percent_separators(self):
        self.assertEqual(format_number(1_234_567.89, 2), "1.234.567,89")
        self.assertEqual(format_number(500_000.0), "500.000")
        self.assertEqual(format_number(0.860000000000001), "0,86")
        self.assertEqual(format_number(12.34567), "12,346")
        self.assertEqual(format_number(-1_234, 0), "-1.234")
        self.assertEqual(format_percent(0.1234, 2), "12,34 %")
        self.assertEqual(f"{es(1_234.5):,.1f}", "1.234,5")
        self.assertEqual(f"{es(0.125):.1%}", "12,5 %")

    def test_numeric_and_percentage_axes_are_localized(self):
        fig, axes = plt.subplots(1, 2)
        axes[0].plot([0, 2_000], [0, 1])
        axes[1].set_ylim(0, 1)
        axes[1].yaxis.set_major_formatter(PercentFormatter(1, decimals=1))
        for axis in axes:
            localize_axis_numbers(axis)
        self.assertEqual(axes[0].xaxis.get_major_formatter()(1_500, 0), "1.500")
        self.assertEqual(axes[1].yaxis.get_major_formatter()(0.125, 0), "12,5 %")
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
