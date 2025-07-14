

import time
import random
import matplotlib.pyplot as plt
from typing import Any, Optional, List
import sys


class DynamicArray:
    """
    Dynamic Array implementation with automatic resizing.
    
    Features:
    - Automatic capacity management
    - Amortized O(1) append operation
    - O(1) access by index
    - O(n) insertion/deletion at arbitrary positions
    """
    
    def __init__(self, initial_capacity: int = 4):
        """Initialize dynamic array with given capacity."""
        self._capacity = initial_capacity
        self._size = 0
        self._data = [None] * self._capacity
        self.operation_count = 0
        
    def __len__(self) -> int:
        """Return the number of elements in the array."""
        return self._size
    
    def __getitem__(self, index: int) -> Any:
        """Access element at given index."""
        self.operation_count += 1
        if not 0 <= index < self._size:
            raise IndexError("Array index out of range")
        return self._data[index]
    
    def __setitem__(self, index: int, value: Any) -> None:
        """Set element at given index."""
        self.operation_count += 1
        if not 0 <= index < self._size:
            raise IndexError("Array index out of range")
        self._data[index] = value
    
    def _resize(self, new_capacity: int) -> None:
        """Resize the internal array to new capacity."""
        old_data = self._data
        self._data = [None] * new_capacity
        self._capacity = new_capacity
        
        # Copy elements to new array
        for i in range(self._size):
            self._data[i] = old_data[i]
            self.operation_count += 1
    
    def append(self, value: Any) -> None:
        """Add element to end of array. Amortized O(1)."""
        if self._size >= self._capacity:
            self._resize(2 * self._capacity)  # Double the capacity
        
        self._data[self._size] = value
        self._size += 1
        self.operation_count += 1
    
    def insert(self, index: int, value: Any) -> None:
        """Insert element at given index. O(n) worst case."""
        if not 0 <= index <= self._size:
            raise IndexError("Insert index out of range")
        
        if self._size >= self._capacity:
            self._resize(2 * self._capacity)
        
        # Shift elements to the right
        for i in range(self._size, index, -1):
            self._data[i] = self._data[i - 1]
            self.operation_count += 1
        
        self._data[index] = value
        self._size += 1
        self.operation_count += 1
    
    def delete(self, index: int) -> Any:
        """Delete and return element at given index. O(n) worst case."""
        if not 0 <= index < self._size:
            raise IndexError("Delete index out of range")
        
        value = self._data[index]
        
        # Shift elements to the left
        for i in range(index, self._size - 1):
            self._data[i] = self._data[i + 1]
            self.operation_count += 1
        
        self._size -= 1
        self.operation_count += 1
        
        # Shrink if size is 1/4 of capacity
        if self._size > 0 and self._size <= self._capacity // 4:
            self._resize(self._capacity // 2)
        
        return value
    
    def search(self, value: Any) -> int:
        """Linear search for value. Return index or -1 if not found."""
        for i in range(self._size):
            self.operation_count += 1
            if self._data[i] == value:
                return i
        return -1
    
    def display(self) -> List[Any]:
        """Return list representation of current elements."""
        return [self._data[i] for i in range(self._size)]


class ArrayStack:
    """
    Stack implementation using dynamic array.
    
    Features:
    - LIFO (Last In, First Out) ordering
    - O(1) amortized push and pop operations
    - O(1) peek operation
    """
    
    def __init__(self):
        """Initialize empty stack."""
        self._array = DynamicArray()
        self.operation_count = 0
    
    def push(self, value: Any) -> None:
        """Push element onto stack. Amortized O(1)."""
        self._array.append(value)
        self.operation_count += 1
    
    def pop(self) -> Any:
        """Pop and return top element. O(1)."""
        if self.is_empty():
            raise IndexError("Pop from empty stack")
        
        self.operation_count += 1
        return self._array.delete(len(self._array) - 1)
    
    def peek(self) -> Any:
        """Return top element without removing it. O(1)."""
        if self.is_empty():
            raise IndexError("Peek from empty stack")
        
        self.operation_count += 1
        return self._array[len(self._array) - 1]
    
    def is_empty(self) -> bool:
        """Check if stack is empty. O(1)."""
        return len(self._array) == 0
    
    def size(self) -> int:
        """Return number of elements in stack. O(1)."""
        return len(self._array)
    
    def display(self) -> List[Any]:
        """Return list representation with top element last."""
        return self._array.display()


class ArrayQueue:
    """
    Queue implementation using circular array.
    
    Features:
    - FIFO (First In, First Out) ordering
    - O(1) enqueue and dequeue operations
    - Circular buffer to avoid shifting elements
    """
    
    def __init__(self, initial_capacity: int = 4):
        """Initialize empty queue with given capacity."""
        self._capacity = initial_capacity
        self._data = [None] * self._capacity
        self._front = 0
        self._size = 0
        self.operation_count = 0
    
    def _resize(self, new_capacity: int) -> None:
        """Resize the queue to new capacity."""
        old_data = self._data
        self._data = [None] * new_capacity
        
        # Copy elements in correct order
        for i in range(self._size):
            self._data[i] = old_data[(self._front + i) % self._capacity]
            self.operation_count += 1
        
        self._front = 0
        self._capacity = new_capacity
    
    def enqueue(self, value: Any) -> None:
        """Add element to rear of queue. Amortized O(1)."""
        if self._size >= self._capacity:
            self._resize(2 * self._capacity)
        
        rear = (self._front + self._size) % self._capacity
        self._data[rear] = value
        self._size += 1
        self.operation_count += 1
    
    def dequeue(self) -> Any:
        """Remove and return front element. O(1)."""
        if self.is_empty():
            raise IndexError("Dequeue from empty queue")
        
        value = self._data[self._front]
        self._data[self._front] = None  # Clear reference
        self._front = (self._front + 1) % self._capacity
        self._size -= 1
        self.operation_count += 1
        
        # Shrink if size is 1/4 of capacity
        if self._size > 0 and self._size <= self._capacity // 4:
            self._resize(self._capacity // 2)
        
        return value
    
    def front(self) -> Any:
        """Return front element without removing it. O(1)."""
        if self.is_empty():
            raise IndexError("Front from empty queue")
        
        self.operation_count += 1
        return self._data[self._front]
    
    def is_empty(self) -> bool:
        """Check if queue is empty. O(1)."""
        return self._size == 0
    
    def size(self) -> int:
        """Return number of elements in queue. O(1)."""
        return self._size
    
    def display(self) -> List[Any]:
        """Return list representation with front element first."""
        result = []
        for i in range(self._size):
            result.append(self._data[(self._front + i) % self._capacity])
        return result


class ListNode:
    """Node class for singly linked list."""
    
    def __init__(self, data: Any = None, next_node: Optional['ListNode'] = None):
        """Initialize node with data and next pointer."""
        self.data = data
        self.next = next_node


class SinglyLinkedList:
    """
    Singly Linked List implementation.
    
    Features:
    - Dynamic size allocation
    - O(1) insertion/deletion at head
    - O(n) insertion/deletion at arbitrary positions
    - O(n) search operation
    """
    
    def __init__(self):
        """Initialize empty linked list."""
        self._head = None
        self._size = 0
        self.operation_count = 0
    
    def insert_head(self, data: Any) -> None:
        """Insert element at head of list. O(1)."""
        new_node = ListNode(data, self._head)
        self._head = new_node
        self._size += 1
        self.operation_count += 1
    
    def insert_tail(self, data: Any) -> None:
        """Insert element at tail of list. O(n)."""
        new_node = ListNode(data)
        
        if not self._head:
            self._head = new_node
        else:
            current = self._head
            while current.next:
                current = current.next
                self.operation_count += 1
            current.next = new_node
        
        self._size += 1
        self.operation_count += 1
    
    def insert_at_position(self, position: int, data: Any) -> None:
        """Insert element at given position. O(n)."""
        if position < 0 or position > self._size:
            raise IndexError("Position out of range")
        
        if position == 0:
            self.insert_head(data)
            return
        
        new_node = ListNode(data)
        current = self._head
        
        # Traverse to position - 1
        for i in range(position - 1):
            current = current.next
            self.operation_count += 1
        
        new_node.next = current.next
        current.next = new_node
        self._size += 1
        self.operation_count += 1
    
    def delete_head(self) -> Any:
        """Delete and return head element. O(1)."""
        if not self._head:
            raise IndexError("Delete from empty list")
        
        data = self._head.data
        self._head = self._head.next
        self._size -= 1
        self.operation_count += 1
        return data
    
    def delete_tail(self) -> Any:
        """Delete and return tail element. O(n)."""
        if not self._head:
            raise IndexError("Delete from empty list")
        
        if not self._head.next:  # Only one element
            data = self._head.data
            self._head = None
            self._size -= 1
            self.operation_count += 1
            return data
        
        # Find second-to-last node
        current = self._head
        while current.next.next:
            current = current.next
            self.operation_count += 1
        
        data = current.next.data
        current.next = None
        self._size -= 1
        self.operation_count += 1
        return data
    
    def delete_at_position(self, position: int) -> Any:
        """Delete and return element at given position. O(n)."""
        if position < 0 or position >= self._size:
            raise IndexError("Position out of range")
        
        if position == 0:
            return self.delete_head()
        
        current = self._head
        
        # Traverse to position - 1
        for i in range(position - 1):
            current = current.next
            self.operation_count += 1
        
        data = current.next.data
        current.next = current.next.next
        self._size -= 1
        self.operation_count += 1
        return data
    
    def search(self, data: Any) -> int:
        """Search for element and return position. O(n)."""
        current = self._head
        position = 0
        
        while current:
            self.operation_count += 1
            if current.data == data:
                return position
            current = current.next
            position += 1
        
        return -1  # Not found
    
    def get(self, position: int) -> Any:
        """Get element at given position. O(n)."""
        if position < 0 or position >= self._size:
            raise IndexError("Position out of range")
        
        current = self._head
        for i in range(position):
            current = current.next
            self.operation_count += 1
        
        return current.data
    
    def size(self) -> int:
        """Return number of elements in list. O(1)."""
        return self._size
    
    def is_empty(self) -> bool:
        """Check if list is empty. O(1)."""
        return self._head is None
    
    def display(self) -> List[Any]:
        """Return list representation of elements. O(n)."""
        result = []
        current = self._head
        while current:
            result.append(current.data)
            current = current.next
        return result


class TreeNode:
    """Node class for binary tree."""
    
    def __init__(self, data: Any = None):
        """Initialize tree node with data."""
        self.data = data
        self.left = None
        self.right = None


class BinaryTree:
    """
    Binary Tree implementation using linked nodes.
    
    Features:
    - Hierarchical data structure
    - Various traversal methods (inorder, preorder, postorder)
    - Basic tree operations
    """
    
    def __init__(self):
        """Initialize empty binary tree."""
        self.root = None
        self.operation_count = 0
    
    def insert(self, data: Any) -> None:
        """Insert data into binary search tree. O(log n) average, O(n) worst."""
        if not self.root:
            self.root = TreeNode(data)
        else:
            self._insert_recursive(self.root, data)
        self.operation_count += 1
    
    def _insert_recursive(self, node: TreeNode, data: Any) -> None:
        """Recursive helper for insertion."""
        self.operation_count += 1
        if data <= node.data:
            if not node.left:
                node.left = TreeNode(data)
            else:
                self._insert_recursive(node.left, data)
        else:
            if not node.right:
                node.right = TreeNode(data)
            else:
                self._insert_recursive(node.right, data)
    
    def search(self, data: Any) -> bool:
        """Search for data in tree. O(log n) average, O(n) worst."""
        return self._search_recursive(self.root, data)
    
    def _search_recursive(self, node: TreeNode, data: Any) -> bool:
        """Recursive helper for search."""
        if not node:
            return False
        
        self.operation_count += 1
        if data == node.data:
            return True
        elif data < node.data:
            return self._search_recursive(node.left, data)
        else:
            return self._search_recursive(node.right, data)
    
    def inorder_traversal(self) -> List[Any]:
        """Return inorder traversal of tree. O(n)."""
        result = []
        self._inorder_recursive(self.root, result)
        return result
    
    def _inorder_recursive(self, node: TreeNode, result: List[Any]) -> None:
        """Recursive helper for inorder traversal."""
        if node:
            self._inorder_recursive(node.left, result)
            result.append(node.data)
            self.operation_count += 1
            self._inorder_recursive(node.right, result)
    
    def height(self) -> int:
        """Return height of tree. O(n)."""
        return self._height_recursive(self.root)
    
    def _height_recursive(self, node: TreeNode) -> int:
        """Recursive helper for height calculation."""
        if not node:
            return -1
        
        self.operation_count += 1
        left_height = self._height_recursive(node.left)
        right_height = self._height_recursive(node.right)
        return 1 + max(left_height, right_height)


class DataStructureAnalyzer:
    """
    Analyzer class for empirical performance testing of data structures.
    """
    
    def __init__(self):
        """Initialize analyzer."""
        self.results = {}
    
    def test_array_operations(self, sizes: List[int]) -> dict:
        """Test dynamic array operations across different sizes."""
        results = {
            'append': {'sizes': sizes, 'times': [], 'operations': []},
            'insert_middle': {'sizes': sizes, 'times': [], 'operations': []},
            'delete_middle': {'sizes': sizes, 'times': [], 'operations': []},
            'search': {'sizes': sizes, 'times': [], 'operations': []}
        }
        
        for size in sizes:
            # Test append operation
            array = DynamicArray()
            array.operation_count = 0
            
            start_time = time.time()
            for i in range(size):
                array.append(i)
            end_time = time.time()
            
            results['append']['times'].append(end_time - start_time)
            results['append']['operations'].append(array.operation_count)
            
            # Test insert at middle
            array.operation_count = 0
            start_time = time.time()
            array.insert(size // 2, -1)
            end_time = time.time()
            
            results['insert_middle']['times'].append(end_time - start_time)
            results['insert_middle']['operations'].append(array.operation_count)
            
            # Test delete at middle
            array.operation_count = 0
            start_time = time.time()
            array.delete(size // 2)
            end_time = time.time()
            
            results['delete_middle']['times'].append(end_time - start_time)
            results['delete_middle']['operations'].append(array.operation_count)
            
            # Test search
            array.operation_count = 0
            start_time = time.time()
            array.search(size - 1)
            end_time = time.time()
            
            results['search']['times'].append(end_time - start_time)
            results['search']['operations'].append(array.operation_count)
        
        return results
    
    def test_stack_queue_operations(self, sizes: List[int]) -> dict:
        """Test stack and queue operations."""
        results = {
            'stack_push_pop': {'sizes': sizes, 'times': [], 'operations': []},
            'queue_enqueue_dequeue': {'sizes': sizes, 'times': [], 'operations': []}
        }
        
        for size in sizes:
            # Test stack operations
            stack = ArrayStack()
            stack.operation_count = 0
            
            start_time = time.time()
            for i in range(size):
                stack.push(i)
            for i in range(size):
                stack.pop()
            end_time = time.time()
            
            results['stack_push_pop']['times'].append(end_time - start_time)
            results['stack_push_pop']['operations'].append(stack.operation_count)
            
            # Test queue operations
            queue = ArrayQueue()
            queue.operation_count = 0
            
            start_time = time.time()
            for i in range(size):
                queue.enqueue(i)
            for i in range(size):
                queue.dequeue()
            end_time = time.time()
            
            results['queue_enqueue_dequeue']['times'].append(end_time - start_time)
            results['queue_enqueue_dequeue']['operations'].append(queue.operation_count)
        
        return results
    
    def test_linked_list_operations(self, sizes: List[int]) -> dict:
        """Test linked list operations."""
        results = {
            'insert_head': {'sizes': sizes, 'times': [], 'operations': []},
            'insert_tail': {'sizes': sizes, 'times': [], 'operations': []},
            'search': {'sizes': sizes, 'times': [], 'operations': []}
        }
        
        for size in sizes:
            # Test head insertion
            linked_list = SinglyLinkedList()
            linked_list.operation_count = 0
            
            start_time = time.time()
            for i in range(size):
                linked_list.insert_head(i)
            end_time = time.time()
            
            results['insert_head']['times'].append(end_time - start_time)
            results['insert_head']['operations'].append(linked_list.operation_count)
            
            # Test tail insertion
            linked_list = SinglyLinkedList()
            linked_list.operation_count = 0
            
            start_time = time.time()
            for i in range(min(size, 1000)):  # Limit for tail insertion
                linked_list.insert_tail(i)
            end_time = time.time()
            
            results['insert_tail']['times'].append(end_time - start_time)
            results['insert_tail']['operations'].append(linked_list.operation_count)
            
            # Test search
            linked_list.operation_count = 0
            start_time = time.time()
            linked_list.search(min(size, 1000) - 1)
            end_time = time.time()
            
            results['search']['times'].append(end_time - start_time)
            results['search']['operations'].append(linked_list.operation_count)
        
        return results
    
    def compare_structures(self) -> None:
        """Compare different data structures for specific operations."""
        print("=== Data Structure Performance Comparison ===\n")
        
        # Test sizes
        sizes = [100, 500, 1000, 2000]
        
        print("Testing Array Operations...")
        array_results = self.test_array_operations(sizes)
        
        print("Testing Stack and Queue Operations...")
        stack_queue_results = self.test_stack_queue_operations(sizes)
        
        print("Testing Linked List Operations...")
        linked_list_results = self.test_linked_list_operations(sizes)
        
        # Display results
        self._display_results(array_results, "Dynamic Array")
        self._display_results(stack_queue_results, "Stack and Queue")
        self._display_results(linked_list_results, "Linked List")
    
    def _display_results(self, results: dict, structure_name: str) -> None:
        """Display performance results in formatted table."""
        print(f"\n{structure_name} Performance Results:")
        print("-" * 60)
        
        for operation, data in results.items():
            print(f"\n{operation.replace('_', ' ').title()} Operation:")
            print("Size\t\tTime (s)\t\tOperations")
            print("-" * 40)
            
            for i, size in enumerate(data['sizes']):
                time_val = data['times'][i]
                ops = data['operations'][i]
                print(f"{size}\t\t{time_val:.6f}\t\t{ops}")


def demonstrate_data_structures():
    """Demonstrate all implemented data structures with examples."""
    print("=== Elementary Data Structures Demonstration ===\n")
    
    # Dynamic Array Demo
    print("1. Dynamic Array Operations:")
    print("-" * 30)
    
    array = DynamicArray()
    
    # Test basic operations
    print("Appending elements 1, 2, 3...")
    for i in [1, 2, 3]:
        array.append(i)
    print(f"Array: {array.display()}")
    
    print("\nInserting 0 at position 0...")
    array.insert(0, 0)
    print(f"Array: {array.display()}")
    
    print("\nDeleting element at position 2...")
    deleted = array.delete(2)
    print(f"Deleted: {deleted}, Array: {array.display()}")
    
    print(f"\nSearching for element 2: position {array.search(2)}")
    
    # Stack Demo
    print("\n\n2. Stack Operations:")
    print("-" * 20)
    
    stack = ArrayStack()
    
    print("Pushing elements 10, 20, 30...")
    for i in [10, 20, 30]:
        stack.push(i)
    print(f"Stack: {stack.display()}")
    
    print(f"\nPeek top element: {stack.peek()}")
    print(f"Pop element: {stack.pop()}")
    print(f"Stack after pop: {stack.display()}")
    
    # Queue Demo
    print("\n\n3. Queue Operations:")
    print("-" * 20)
    
    queue = ArrayQueue()
    
    print("Enqueueing elements 100, 200, 300...")
    for i in [100, 200, 300]:
        queue.enqueue(i)
    print(f"Queue: {queue.display()}")
    
    print(f"\nFront element: {queue.front()}")
    print(f"Dequeue element: {queue.dequeue()}")
    print(f"Queue after dequeue: {queue.display()}")
    
    # Linked List Demo
    print("\n\n4. Linked List Operations:")
    print("-" * 25)
    
    linked_list = SinglyLinkedList()
    
    print("Inserting at head: 3, 2, 1...")
    for i in [3, 2, 1]:
        linked_list.insert_head(i)
    print(f"List: {linked_list.display()}")
    
    print("\nInserting at tail: 4...")
    linked_list.insert_tail(4)
    print(f"List: {linked_list.display()}")
    
    print("\nInserting 2.5 at position 3...")
    linked_list.insert_at_position(3, 2.5)
    print(f"List: {linked_list.display()}")
    
    print(f"\nSearching for element 2.5: position {linked_list.search(2.5)}")
    
    print(f"\nDeleting head: {linked_list.delete_head()}")
    print(f"List after deletion: {linked_list.display()}")
    
    # Binary Tree Demo
    print("\n\n5. Binary Tree Operations:")
    print("-" * 25)
    
    tree = BinaryTree()
    
    print("Inserting elements: 5, 3, 7, 2, 4, 6, 8...")
    for val in [5, 3, 7, 2, 4, 6, 8]:
        tree.insert(val)
    
    print(f"Inorder traversal: {tree.inorder_traversal()}")
    print(f"Tree height: {tree.height()}")
    print(f"Search for 4: {tree.search(4)}")
    print(f"Search for 9: {tree.search(9)}")


def run_performance_analysis():
    """Run comprehensive performance analysis."""
    print("\n=== Performance Analysis ===")
    
    analyzer = DataStructureAnalyzer()
    analyzer.compare_structures()


if __name__ == "__main__":
    print("Elementary Data Structures Implementation")
    print("=" * 50)
    
    # Run demonstrations
    demonstrate_data_structures()
    
    # Run performance analysis
    run_performance_analysis()
    
    print("\n=== Analysis Complete ===")
    print("\nKey Findings:")
    print("1. Array operations: O(1) access, O(n) insertion/deletion")
    print("2. Stack/Queue: O(1) push/pop/enqueue/dequeue operations")
    print("3. Linked List: O(1) head operations, O(n) arbitrary position")
    print("4. Binary Tree: O(log n) average, O(n) worst for search/insert")
    
    print("\nFor detailed analysis, see the accompanying documentation.")