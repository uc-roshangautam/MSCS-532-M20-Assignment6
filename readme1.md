Selection Algorithms Implementation

Overview
Implementation of two selection algorithms for finding the k-th smallest element:
1.	Median of Medians - O(n) worst-case time complexity
2.	Randomized Quickselect - O(n) expected time complexity
Files
├── selection_algorithms.py       # Main implementation
└── README.md                     # This file

Requirements
•	Python 3.7+
•	NumPy: pip install numpy
•	Matplotlib: pip install matplotlib

Quick Start
Run Complete Analysis
python selection_algorithms.py
Run Tests
python test_selection_algorithms.py
Basic Usage
from selection_algorithms import SelectionAlgorithms

selector = SelectionAlgorithms()
arr = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]

# Find 5th smallest element
result_mom = selector.median_of_medians_select(arr, 5)
result_qs = selector.randomized_quickselect(arr, 5)

print(f"Results: MoM={result_mom}, QS={result_qs}")

Algorithm Summary
Algorithm	Time Complexity	Space	Best For
Median of Medians	O(n) worst-case	O(log n)	Guaranteed performance
Randomized Quickselect	O(n) expected	O(log n)	Average performance

Output
Running the main script produces:
•	Algorithm demonstrations with test cases
•	Performance analysis across different data types
•	Comparison plots (saved as PNG files)
•	Summary statistics

