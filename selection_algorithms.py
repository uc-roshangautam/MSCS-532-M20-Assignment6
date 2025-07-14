import random
import time
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple


class SelectionAlgorithms:
    """
    Implementation of selection algorithms for finding the k-th smallest element.
    Includes both deterministic (Median of Medians) and randomized (Quickselect) approaches.
    """
    
    def __init__(self):
        self.comparison_count = 0
    
    def reset_comparisons(self):
        """Reset the comparison counter for analysis."""
        self.comparison_count = 0
    
    def compare(self, a, b):
        """Compare two elements and increment counter."""
        self.comparison_count += 1
        return a < b
    
    def median_of_medians_select(self, arr: List[int], k: int) -> int:
        """
        Deterministic selection algorithm using Median of Medians.
        Guarantees O(n) worst-case time complexity.
        
        Args:
            arr: List of integers
            k: Find the k-th smallest element (1-indexed)
            
        Returns:
            The k-th smallest element
        """
        if not arr or k <= 0 or k > len(arr):
            raise ValueError("Invalid input parameters")
        
        return self._mom_select_helper(arr.copy(), k - 1)  # Convert to 0-indexed
    
    def _mom_select_helper(self, arr: List[int], k: int) -> int:
        """Helper function for median of medians selection."""
        n = len(arr)
        
        # Base case: small arrays
        if n <= 5:
            arr.sort()
            return arr[k]
        
        # Step 1: Divide into groups of 5 and find medians
        medians = []
        for i in range(0, n, 5):
            group = arr[i:i+5]
            group.sort()
            medians.append(group[len(group) // 2])
        
        # Step 2: Recursively find median of medians
        pivot = self._mom_select_helper(medians, len(medians) // 2)
        
        # Step 3: Partition around the pivot
        left, equal, right = self._partition_three_way(arr, pivot)
        
        # Step 4: Recursively search in appropriate partition
        if k < len(left):
            return self._mom_select_helper(left, k)
        elif k < len(left) + len(equal):
            return pivot
        else:
            return self._mom_select_helper(right, k - len(left) - len(equal))
    
    def _partition_three_way(self, arr: List[int], pivot: int) -> Tuple[List[int], List[int], List[int]]:
        """Three-way partition: elements < pivot, = pivot, > pivot."""
        left, equal, right = [], [], []
        
        for element in arr:
            if self.compare(element, pivot):
                left.append(element)
            elif element == pivot:
                equal.append(element)
            else:
                right.append(element)
        
        return left, equal, right
    
    def randomized_quickselect(self, arr: List[int], k: int) -> int:
        """
        Randomized selection algorithm (Quickselect).
        Expected O(n) time complexity.
        
        Args:
            arr: List of integers
            k: Find the k-th smallest element (1-indexed)
            
        Returns:
            The k-th smallest element
        """
        if not arr or k <= 0 or k > len(arr):
            raise ValueError("Invalid input parameters")
        
        return self._quickselect_helper(arr.copy(), 0, len(arr) - 1, k - 1)
    
    def _quickselect_helper(self, arr: List[int], left: int, right: int, k: int) -> int:
        """Helper function for randomized quickselect."""
        if left == right:
            return arr[left]
        
        # Randomly choose pivot
        pivot_index = random.randint(left, right)
        
        # Partition and get the final position of pivot
        pivot_index = self._partition(arr, left, right, pivot_index)
        
        # Recursively search in appropriate partition
        if k == pivot_index:
            return arr[k]
        elif k < pivot_index:
            return self._quickselect_helper(arr, left, pivot_index - 1, k)
        else:
            return self._quickselect_helper(arr, pivot_index + 1, right, k)
    
    def _partition(self, arr: List[int], left: int, right: int, pivot_index: int) -> int:
        """Partition array around pivot, return final pivot position."""
        pivot_value = arr[pivot_index]
        
        # Move pivot to end
        arr[pivot_index], arr[right] = arr[right], arr[pivot_index]
        
        store_index = left
        for i in range(left, right):
            if self.compare(arr[i], pivot_value):
                arr[store_index], arr[i] = arr[i], arr[store_index]
                store_index += 1
        
        # Move pivot to its final position
        arr[right], arr[store_index] = arr[store_index], arr[right]
        return store_index


class PerformanceAnalyzer:
    """Class for analyzing and comparing the performance of selection algorithms."""
    
    def __init__(self):
        self.selector = SelectionAlgorithms()
    
    def generate_test_data(self, size: int, data_type: str) -> List[int]:
        """Generate test data of different distributions."""
        if data_type == "random":
            return [random.randint(1, size * 10) for _ in range(size)]
        elif data_type == "sorted":
            return list(range(1, size + 1))
        elif data_type == "reverse_sorted":
            return list(range(size, 0, -1))
        elif data_type == "duplicates":
            return [random.randint(1, size // 10) for _ in range(size)]
        else:
            raise ValueError("Unknown data type")
    
    def measure_performance(self, algorithm_func, arr: List[int], k: int) -> Tuple[float, int]:
        """Measure execution time and comparison count for an algorithm."""
        self.selector.reset_comparisons()
        
        start_time = time.perf_counter()
        result = algorithm_func(arr, k)
        end_time = time.perf_counter()
        
        execution_time = end_time - start_time
        comparisons = self.selector.comparison_count
        
        return execution_time, comparisons
    
    def run_empirical_analysis(self, sizes: List[int], data_types: List[str], trials: int = 5):
        """Run comprehensive empirical analysis comparing both algorithms."""
        results = {
            'sizes': sizes,
            'mom_times': {dt: [] for dt in data_types},
            'quickselect_times': {dt: [] for dt in data_types},
            'mom_comparisons': {dt: [] for dt in data_types},
            'quickselect_comparisons': {dt: [] for dt in data_types}
        }
        
        for size in sizes:
            print(f"Testing size: {size}")
            
            for data_type in data_types:
                mom_times, qs_times = [], []
                mom_comps, qs_comps = [], []
                
                for trial in range(trials):
                    # Generate test data
                    arr = self.generate_test_data(size, data_type)
                    k = size // 2  # Find median
                    
                    # Test Median of Medians
                    try:
                        time_mom, comp_mom = self.measure_performance(
                            self.selector.median_of_medians_select, arr, k
                        )
                        mom_times.append(time_mom)
                        mom_comps.append(comp_mom)
                    except Exception as e:
                        print(f"MoM failed for size {size}, type {data_type}: {e}")
                        continue
                    
                    # Test Randomized Quickselect
                    try:
                        time_qs, comp_qs = self.measure_performance(
                            self.selector.randomized_quickselect, arr, k
                        )
                        qs_times.append(time_qs)
                        qs_comps.append(comp_qs)
                    except Exception as e:
                        print(f"Quickselect failed for size {size}, type {data_type}: {e}")
                        continue
                
                # Store average results
                if mom_times and qs_times:
                    results['mom_times'][data_type].append(np.mean(mom_times))
                    results['quickselect_times'][data_type].append(np.mean(qs_times))
                    results['mom_comparisons'][data_type].append(np.mean(mom_comps))
                    results['quickselect_comparisons'][data_type].append(np.mean(qs_comps))
                else:
                    results['mom_times'][data_type].append(0)
                    results['quickselect_times'][data_type].append(0)
                    results['mom_comparisons'][data_type].append(0)
                    results['quickselect_comparisons'][data_type].append(0)
        
        return results
    
    def plot_results(self, results, save_plots=False):
        """Create visualizations of the performance analysis."""
        sizes = results['sizes']
        data_types = ['random', 'sorted', 'reverse_sorted', 'duplicates']
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Selection Algorithms Performance Comparison', fontsize=16)
        
        # Plot execution times
        for i, data_type in enumerate(data_types):
            ax = axes[i // 2, i % 2]
            
            mom_times = results['mom_times'][data_type]
            qs_times = results['quickselect_times'][data_type]
            
            ax.plot(sizes, mom_times, 'b-o', label='Median of Medians', linewidth=2)
            ax.plot(sizes, qs_times, 'r-s', label='Randomized Quickselect', linewidth=2)
            
            ax.set_xlabel('Input Size')
            ax.set_ylabel('Execution Time (seconds)')
            ax.set_title(f'Performance on {data_type.replace("_", " ").title()} Data')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xscale('log')
            ax.set_yscale('log')
        
        plt.tight_layout()
        if save_plots:
            plt.savefig('selection_algorithms_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot comparison counts
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Selection Algorithms Comparison Count Analysis', fontsize=16)
        
        for i, data_type in enumerate(data_types):
            ax = axes[i // 2, i % 2]
            
            mom_comps = results['mom_comparisons'][data_type]
            qs_comps = results['quickselect_comparisons'][data_type]
            
            ax.plot(sizes, mom_comps, 'b-o', label='Median of Medians', linewidth=2)
            ax.plot(sizes, qs_comps, 'r-s', label='Randomized Quickselect', linewidth=2)
            
            ax.set_xlabel('Input Size')
            ax.set_ylabel('Number of Comparisons')
            ax.set_title(f'Comparisons on {data_type.replace("_", " ").title()} Data')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xscale('log')
            ax.set_yscale('log')
        
        plt.tight_layout()
        if save_plots:
            plt.savefig('selection_algorithms_comparisons.png', dpi=300, bbox_inches='tight')
        plt.show()


def demonstrate_algorithms():
    """Demonstrate the selection algorithms with example usage."""
    print("=" * 60)
    print("SELECTION ALGORITHMS DEMONSTRATION")
    print("=" * 60)
    
    selector = SelectionAlgorithms()
    
    # Test cases
    test_arrays = [
        [3, 1, 4, 1, 5, 9, 2, 6, 5, 3],
        [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        [1, 1, 1, 1, 1],
        [5, 2, 8, 1, 9, 3, 7, 4, 6]
    ]
    
    for i, arr in enumerate(test_arrays, 1):
        print(f"\nTest Case {i}: {arr}")
        print(f"Array length: {len(arr)}")
        
        for k in [1, len(arr)//2, len(arr)]:
            try:
                # Test Median of Medians
                selector.reset_comparisons()
                mom_result = selector.median_of_medians_select(arr, k)
                mom_comparisons = selector.comparison_count
                
                # Test Randomized Quickselect
                selector.reset_comparisons()
                qs_result = selector.randomized_quickselect(arr, k)
                qs_comparisons = selector.comparison_count
                
                print(f"  k={k}: MoM={mom_result} ({mom_comparisons} comp), "
                      f"QS={qs_result} ({qs_comparisons} comp)")
                
                # Verify correctness
                sorted_arr = sorted(arr)
                expected = sorted_arr[k-1]
                if mom_result != expected or qs_result != expected:
                    print(f"    ERROR: Expected {expected}")
                
            except Exception as e:
                print(f"  k={k}: Error - {e}")


def main():
    """Main function to run the complete analysis."""
    print("Starting Selection Algorithms Analysis...")
    
    # Demonstrate basic functionality
    demonstrate_algorithms()
    
    # Run empirical analysis
    print("\n" + "=" * 60)
    print("EMPIRICAL PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    analyzer = PerformanceAnalyzer()
    
    # Test with different sizes and data types
    sizes = [100, 500, 1000, 2000, 5000]
    data_types = ['random', 'sorted', 'reverse_sorted', 'duplicates']
    
    print("Running empirical analysis (this may take a few minutes)...")
    results = analyzer.run_empirical_analysis(sizes, data_types, trials=3)
    
    # Plot results
    analyzer.plot_results(results, save_plots=True)
    
    # Print summary statistics
    print("\nSUMMARY STATISTICS:")
    print("-" * 40)
    
    for data_type in data_types:
        print(f"\n{data_type.replace('_', ' ').title()} Data:")
        
        mom_times = results['mom_times'][data_type]
        qs_times = results['quickselect_times'][data_type]
        
        if mom_times and qs_times:
            avg_mom_time = np.mean(mom_times)
            avg_qs_time = np.mean(qs_times)
            
            print(f"  Average MoM time: {avg_mom_time:.6f} seconds")
            print(f"  Average QS time: {avg_qs_time:.6f} seconds")
            print(f"  Speedup ratio: {avg_mom_time / avg_qs_time:.2f}x")


if __name__ == "__main__":
    main()