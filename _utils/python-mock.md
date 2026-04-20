### **Coding Questions (31)**

| # | Question | Topic |
|---|----------|-------|
| 1 | Fibonacci Series (Iterative) | Loops |
| 2 | Palindrome Check | String |
| 3 | Factorial (Recursive) | Recursion |
| 4 | Prime Number Check | Math |
| 5 | Reverse a String | String |
| 6 | Check Armstrong Number | Math |
| 7 | Find Maximum in List | List |
| 8 | Remove Duplicates from List | List |
| 9 | Count Vowels in String | String |
| 10 | Binary Search | Searching |
| 11 | Bubble Sort | Sorting |
| 12 | Find GCD (Euclidean Algorithm) | Math |
| 13 | Check Anagram | String |
| 14 | Find Second Largest Number | List |
| 15 | Fibonacci (Recursive) | Recursion |
| 16 | Sum of Digits | Math |
| 17 | Check Leap Year | Conditional |
| 18 | Pattern Printing (Triangle) | Nested Loops |
| 19 | Merge Two Sorted Lists | List |
| 20 | Find Missing Number | Math |
| 21 | Count Character Frequency | Dictionary |
| 22 | Lambda Function - Square Numbers | Lambda |
| 23 | Map with Lambda - Double List | Map |
| 24 | Filter with Lambda - Get Even | Filter |
| 25 | Reduce with Lambda - Find Product | Reduce |
| 26 | Map with Multiple Lists | Map |
| 27 | Filter Strings by Length | Filter |
| 28 | Reduce to Find Maximum | Reduce |
| 29 | Exception Handling - Division | Exception |
| 30 | Multiple Exception Handling | Exception |
| 31 | Custom Exception | Exception |

---

### **Quick Reference: Syntax Examples**

```python
# Lambda
square = lambda x: x**2

# Map
list(map(lambda x: x*2, [1,2,3]))

# Filter
list(filter(lambda x: x>0, [-1,2,-3,4]))

# Reduce
from functools import reduce
reduce(lambda a,b: a+b, [1,2,3,4])

# Exception Handling Template
try:
    # risky code
except SpecificError:
    # handle error
else:
    # if no error
finally:
    # always execute
```


## **Theory Questions**

### 1. Define Python.
**Answer:** Python is a high-level, interpreted, interactive, and object-oriented scripting language. It is designed to be highly readable, using English keywords frequently where other languages use punctuation. It supports multiple programming paradigms, including procedural, object-oriented, and functional programming.

### 2. What are the different data types in Python?
**Answer:** Python has several built-in data types:
- **Numeric:** `int`, `float`, `complex`
- **Sequence:** `list`, `tuple`, `range`
- **Text:** `str`
- **Mapping:** `dict`
- **Set:** `set`, `frozenset`
- **Boolean:** `bool`
- **Binary:** `bytes`, `bytearray`, `memoryview`

### 3. What are the characteristics of Object-Oriented Programming (OOP) in Python?
**Answer:** The four main OOP characteristics are:
1. **Encapsulation:** Bundling data and methods within a class, hiding internal details.
2. **Abstraction:** Hiding complex implementation and showing only essential features.
3. **Inheritance:** Creating a new class using an existing class's properties.
4. **Polymorphism:** Using a single interface with different underlying forms (e.g., method overriding).

### 4. Differentiate between `list` and `tuple`.
**Answer:**
| List | Tuple |
|------|-------|
| Mutable (can change) | Immutable (cannot change) |
| Slower performance | Faster performance |
| Uses `[]` brackets | Uses `()` parentheses |
| More methods (append, remove, etc.) | Fewer methods |

### 5. Difference between User-Defined Function (UDF) and Lambda Function.
**Answer:**

| Feature | User-Defined Function (UDF) | Lambda Function |
|---------|----------------------------|-----------------|
| **Definition** | Defined using `def` keyword | Defined using `lambda` keyword |
| **Name** | Has a function name | Anonymous (no name) |
| **Body** | Can have multiple statements | Single expression only |
| **Return** | Explicit `return` statement | Implicitly returns expression |
| **Use Case** | Complex logic, multiple lines | Simple one-liner operations |
| **Example** | `def add(x,y): return x+y` | `lambda x,y: x+y` |

### 6. What is Exception Handling in Python? Explain with keywords.
**Answer:** Exception handling manages runtime errors gracefully without crashing the program.

**Keywords:**
- **`try`** : Block that may raise an exception
- **`except`** : Catches and handles specific exceptions
- **`else`** : Executes if no exception occurs
- **`finally`** : Always executes (cleanup code)
- **`raise`** : Manually trigger an exception

**Example:**
```python
try:
    x = int(input("Enter number: "))
    result = 10 / x
except ValueError:
    print("Invalid number!")
except ZeroDivisionError:
    print("Cannot divide by zero!")
else:
    print(f"Result: {result}")
finally:
    print("Execution complete")
```

---

## **Coding Questions (21) - With Code & Explanation**

---

### 1. Fibonacci Series (Iterative)
**Question:** Generate Fibonacci series up to n terms.

```python
def fibonacci(n):
    a, b = 0, 1
    result = []
    for _ in range(n):
        result.append(a)
        a, b = b, a + b
    return result

# Example
print(fibonacci(10))  # [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
```
**Explanation:** Uses two variables to track previous two numbers. Updates them in each iteration to generate next number. Time complexity: O(n).

---

### 2. Palindrome Check (String)
**Question:** Check if a string is palindrome.

```python
def is_palindrome(s):
    s = s.lower().replace(" ", "")
    return s == s[::-1]

# Example
print(is_palindrome("A man a plan a canal panama"))  # True
print(is_palindrome("hello"))  # False
```
**Explanation:** Converts to lowercase, removes spaces, then compares string with its reverse.

---

### 3. Factorial (Recursive)
**Question:** Find factorial of a number.

```python
def factorial(n):
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)

# Example
print(factorial(5))  # 120
```
**Explanation:** Base case returns 1 for 0! and 1!. Recursive case multiplies n with factorial of n-1.

---

### 4. Prime Number Check
**Question:** Check if a number is prime.

```python
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True

# Example
print(is_prime(17))  # True
print(is_prime(20))  # False
```
**Explanation:** Checks divisibility only up to square root of n for efficiency.

---

### 5. Reverse a String
**Question:** Reverse a string without using slicing.

```python
def reverse_string(s):
    reversed_str = ""
    for char in s:
        reversed_str = char + reversed_str
    return reversed_str

# Example
print(reverse_string("Python"))  # nohtyP
```
**Explanation:** Iterates through each character and prepends it to result string.

---

### 6. Check Armstrong Number
**Question:** Check if number is Armstrong (sum of cubes of digits equals number).

```python
def is_armstrong(n):
    digits = str(n)
    num_digits = len(digits)
    total = sum(int(d) ** num_digits for d in digits)
    return total == n

# Example
print(is_armstrong(153))  # True (1³+5³+3³=153)
print(is_armstrong(123))  # False
```
**Explanation:** Converts number to string, raises each digit to power of total digits, sums and compares.

---

### 7. Find Maximum in List
**Question:** Find maximum element without using max().

```python
def find_max(lst):
    if not lst:
        return None
    max_val = lst[0]
    for num in lst:
        if num > max_val:
            max_val = num
    return max_val

# Example
print(find_max([3, 7, 2, 9, 1]))  # 9
```

---

### 8. Remove Duplicates from List
**Question:** Remove duplicates while preserving order.

```python
def remove_duplicates(lst):
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

# Example
print(remove_duplicates([1, 2, 2, 3, 4, 3, 5]))  # [1, 2, 3, 4, 5]
```

---

### 9. Count Vowels in String
**Question:** Count vowels (a, e, i, o, u) in string.

```python
def count_vowels(s):
    vowels = set('aeiouAEIOU')
    return sum(1 for char in s if char in vowels)

# Example
print(count_vowels("Hello World"))  # 3
```

---

### 10. Binary Search
**Question:** Implement binary search on sorted list.

```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

# Example
arr = [1, 3, 5, 7, 9, 11]
print(binary_search(arr, 7))  # 3
```

---

### 11. Bubble Sort
**Question:** Sort list using bubble sort.

```python
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:
            break
    return arr

# Example
print(bubble_sort([64, 34, 25, 12, 22, 11, 90]))
```

---

### 12. Find GCD (Euclidean Algorithm)
**Question:** Find greatest common divisor.

```python
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

# Example
print(gcd(48, 18))  # 6
```

---

### 13. Check Anagram
**Question:** Check if two strings are anagrams.

```python
def is_anagram(s1, s2):
    s1 = s1.replace(" ", "").lower()
    s2 = s2.replace(" ", "").lower()
    return sorted(s1) == sorted(s2)

# Example
print(is_anagram("listen", "silent"))  # True
```

---

### 14. Find Second Largest Number
**Question:** Find second largest in list.

```python
def second_largest(lst):
    unique_nums = list(set(lst))
    if len(unique_nums) < 2:
        return None
    unique_nums.sort()
    return unique_nums[-2]

# Example
print(second_largest([10, 20, 4, 45, 99, 99]))  # 45
```

---

### 15. Fibonacci (Recursive)
**Question:** Get nth Fibonacci number recursively.

```python
def fib_recursive(n):
    if n <= 1:
        return n
    return fib_recursive(n - 1) + fib_recursive(n - 2)

# Example
print(fib_recursive(7))  # 13
```
**Note:** Inefficient for large n (exponential time).

---

### 16. Sum of Digits
**Question:** Sum all digits in a number.

```python
def sum_of_digits(n):
    n = abs(n)
    total = 0
    while n > 0:
        total += n % 10
        n //= 10
    return total

# Example
print(sum_of_digits(12345))  # 15
```

---

### 17. Check Leap Year
**Question:** Check if year is leap year.

```python
def is_leap_year(year):
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

# Example
print(is_leap_year(2024))  # True
print(is_leap_year(2023))  # False
```

---

### 18. Pattern Printing (Triangle)
**Question:** Print right-angled triangle of stars.

```python
def print_triangle(n):
    for i in range(1, n + 1):
        print('*' * i)

# Example
print_triangle(5)
# Output:
# *
# **
# ***
# ****
# *****
```

---

### 19. Merge Two Sorted Lists
**Question:** Merge two sorted lists into one sorted list.

```python
def merge_sorted(list1, list2):
    result = []
    i = j = 0
    while i < len(list1) and j < len(list2):
        if list1[i] < list2[j]:
            result.append(list1[i])
            i += 1
        else:
            result.append(list2[j])
            j += 1
    result.extend(list1[i:])
    result.extend(list2[j:])
    return result

# Example
print(merge_sorted([1, 3, 5], [2, 4, 6]))  # [1, 2, 3, 4, 5, 6]
```

---

### 20. Find Missing Number
**Question:** Find missing number from 1 to n.

```python
def find_missing(arr, n):
    expected_sum = n * (n + 1) // 2
    actual_sum = sum(arr)
    return expected_sum - actual_sum

# Example
print(find_missing([1, 2, 4, 5, 6], 6))  # 3
```

---

### 21. Count Character Frequency
**Question:** Count frequency of each character in string.

```python
def char_frequency(s):
    freq = {}
    for char in s:
        freq[char] = freq.get(char, 0) + 1
    return freq

# Example
print(char_frequency("hello"))
# {'h': 1, 'e': 1, 'l': 2, 'o': 1}
```

---

### 22. Lambda Function - Square Numbers
**Question:** Create a lambda function to square a number.

```python
# Lambda function
square = lambda x: x ** 2

# Example
print(square(5))  # 25
print(square(10))  # 100
```
**Explanation:** Lambda takes one argument x and returns x². No `return` keyword needed.

---

### 23. Map with Lambda - Double List Elements
**Question:** Use map() and lambda to double all elements in a list.

```python
numbers = [1, 2, 3, 4, 5]
doubled = list(map(lambda x: x * 2, numbers))

print(doubled)  # [2, 4, 6, 8, 10]
```
**Explanation:** `map()` applies lambda function to each element. Returns map object, converted to list.

---

### 24. Filter with Lambda - Get Even Numbers
**Question:** Use filter() and lambda to get even numbers from list.

```python
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
evens = list(filter(lambda x: x % 2 == 0, numbers))

print(evens)  # [2, 4, 6, 8, 10]
```
**Explanation:** `filter()` keeps elements where lambda returns True.

---

### 25. Reduce with Lambda - Find Product
**Question:** Use reduce() and lambda to find product of all list elements.

```python
from functools import reduce

numbers = [1, 2, 3, 4, 5]
product = reduce(lambda x, y: x * y, numbers)

print(product)  # 120 (1*2*3*4*5)
```
**Explanation:** `reduce()` cumulatively applies lambda to pairs of elements.

---

### 26. Map with Multiple Lists
**Question:** Use map() to add corresponding elements from two lists.

```python
list1 = [1, 2, 3, 4]
list2 = [5, 6, 7, 8]
sums = list(map(lambda x, y: x + y, list1, list2))

print(sums)  # [6, 8, 10, 12]
```
**Explanation:** Lambda takes two arguments. Map processes multiple lists in parallel.

---

### 27. Filter Strings by Length
**Question:** Use filter() to get strings longer than 3 characters.

```python
words = ["cat", "elephant", "dog", "tiger", "bird"]
long_words = list(filter(lambda w: len(w) > 3, words))

print(long_words)  # ['elephant', 'tiger', 'bird']
```

---

### 28. Reduce to Find Maximum
**Question:** Use reduce() to find maximum number in list.

```python
from functools import reduce

numbers = [42, 17, 89, 3, 56, 23]
max_num = reduce(lambda a, b: a if a > b else b, numbers)

print(max_num)  # 89
```

---

### 29. Exception Handling - Division
**Question:** Write division program with exception handling.

```python
def safe_divide(a, b):
    try:
        result = a / b
        return result
    except ZeroDivisionError:
        return "Error: Cannot divide by zero!"
    except TypeError:
        return "Error: Please provide numbers!"
    except Exception as e:
        return f"Unexpected error: {e}"

# Examples
print(safe_divide(10, 2))    # 5.0
print(safe_divide(10, 0))    # Error: Cannot divide by zero!
print(safe_divide(10, "2"))  # Error: Please provide numbers!
```

---

### 30. Multiple Exception Handling
**Question:** Handle multiple exception types when converting and processing input.

```python
def process_input():
    try:
        user_input = input("Enter numbers separated by comma: ")
        numbers = [int(x.strip()) for x in user_input.split(",")]
        average = sum(numbers) / len(numbers)
        print(f"Average: {average}")
    
    except ValueError:
        print("Error: Please enter valid numbers only!")
    except ZeroDivisionError:
        print("Error: Empty list provided!")
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    finally:
        print("Process completed")

# process_input()  # Uncomment to test
```

---

### 31. Custom Exception
**Question:** Create and raise custom exception for age validation.

```python
class AgeTooLowError(Exception):
    """Custom exception for age below minimum"""
    pass

def validate_age(age):
    try:
        if age < 18:
            raise AgeTooLowError(f"Age {age} is below minimum 18")
        else:
            print(f"Age {age} is valid")
    except AgeTooLowError as e:
        print(f"Custom Exception: {e}")

# Examples
validate_age(25)  # Age 25 is valid
validate_age(15)  # Custom Exception: Age 15 is below minimum 18
```

---

## **Summary Table**

| Type | Count |
|------|-------|
| Theory Questions | 6 |
| Coding Questions | 25 |
| **Total** | **31** |
