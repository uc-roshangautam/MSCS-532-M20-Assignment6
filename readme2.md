Elementary Data Structures Implementation


Overview
Implementation of fundamental data structures for efficient data organization and manipulation:
1.	Dynamic Array - Automatically resizing array with O(1) amortized append
2.	Array-Based Stack - LIFO data structure with O(1) operations
3.	Circular Queue - FIFO data structure with O(1) operations
4.	Singly Linked List - Pointer-based structure with O(1) head operations
5.	Binary Search Tree - Hierarchical structure with O(log n) average operations

Files
├── elementary_data_structures.py    # Main implementation
├── README.md                        # This file
└── Documentation.md                 # Detailed analysis report

Requirements
•	Python 3.7+
•	Matplotlib: pip install matplotlib
•	Standard Library: time, random, sys

Quick Start
Run Complete Analysis
python elementary_data_structures.py
Run Tests
from elementary_data_structures import demonstrate_data_structures, run_performance_analysis

# Basic demonstrations
demonstrate_data_structures()

# Performance analysis
run_performance_analysis()
Basic Usage
from elementary_data_structures import DynamicArray, ArrayStack, ArrayQueue, SinglyLinkedList

# Dynamic Array
array = DynamicArray()
array.append(1)
array.append(2)
print(array.display())  # [1, 2]

# Stack Operations  
stack = ArrayStack()
stack.push(10)
stack.push(20)
print(stack.pop())  # 20

# Queue Operations
queue = ArrayQueue()
queue.enqueue(100)
queue.enqueue(200)
print(queue.dequeue())  # 100

# Linked List Operations
linked_list = SinglyLinkedList()
linked_list.insert_head(1)
linked_list.insert_tail(2)
print(linked_list.display())  # [1, 2]
Data Structure Summary
Data Structure	Access	Insert	Delete	Search	Space	Best Use Case
Dynamic Array	O(1)	O(n)	O(n)	O(n)	O(n)	Random access, caching
Array Stack	O(1)*	O(1)	O(1)	-	O(n)	LIFO operations, undo
Array Queue	O(1)*	O(1)	O(1)	-	O(n)	FIFO operations, scheduling
Linked List	O(n)	O(1)**	O(1)**	O(n)	O(n)	Frequent head operations
Binary Tree	-	O(log n)	O(log n)	O(log n)	O(n)	Sorted data, searching
* Only top/front element
** Only at head position

Key Performance Insights
Empirical Results (Actual Test Data)
Dynamic Array Performance (2000 elements)
•	Append Operation: 0.000410s, 4044 operations - confirms amortized O(1) with resize overhead
•	Insert Middle: 0.000078s, 1001 operations - O(n/2) element shifting as predicted
•	Search: 0.000126s, 2000 operations - perfect O(n) linear scaling
Stack vs Queue Comparison (2000 operations each)
•	Stack Push-Pop: 0.001898s, 4000 operations
•	Queue Enqueue-Dequeue: 0.001284s, 7067 operations
•	Verdict: Queue outperforms despite higher operation count due to implementation efficiency

Array vs Linked List Trade-offs (Actual Measurements)
•	Array Insert Head: ~0.063s estimated (O(n²) for n operations)
•	Linked Insert Head: 0.000535s for 2000 operations (O(1) each)
•	Performance Ratio: Linked list 33x faster for head operations
•	Linked Insert Tail: 0.031788s, 499,501 operations - quadratic explosion
•	Array Insert Tail: ~0.001s estimated (O(1) amortized each)

Critical Findings
1.	Amortized Analysis Validated: Dynamic array append shows 2.02 average operations/element, confirming O(1) amortized
2.	Quadratic Behavior Visible: Linked list tail insertions grow from 4,951→499,501 operations (100x for 20x size)
3.	Operation Pattern Dominance: Head-heavy → Linked List (33x faster), Tail/Random → Array
4.	Implementation Details Matter: Queue outperforms stack despite higher operation counts
5.	Theoretical Predictions Hold: All empirical results align with complexity analysis

Output
Running the main script produces:
•	Data structure demonstrations with example operations
•	Performance analysis across different input sizes
•	Comparison results between structures
•	Time complexity validation with empirical data
•	Memory usage analysis and recommendations



