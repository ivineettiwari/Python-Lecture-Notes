# Python Question Bank with Answers (Questions)

## Table of Contents
1. [Basic Python (30 Questions)](#basic-python)
2. [Data Structures (30 Questions)](#data-structures)
3. [Functions (20 Questions)](#functions)
4. [Object-Oriented Programming (20 Questions)](#oop)
5. [File Handling (15 Questions)](#file-handling)
6. [Error Handling (15 Questions)](#error-handling)
7. [Modules and Packages (15 Questions)](#modules-packages)
8. [Python Standard Library (15 Questions)](#standard-library)
9. [Intermediate Python (20 Questions)](#intermediate-python)
10. [Advanced Python (20 Questions)](#advanced-python)

---

## <a name="basic-python"></a>Basic Python (30 Questions)

1. **What is Python and what are its key features?**
   - Answer: Python is a high-level, interpreted, interactive and object-oriented scripting language. Key features include:
     - Easy to learn and read
     - Interpreted language
     - Dynamically typed
     - Cross-platform
     - Extensive standard library
     - Supports multiple programming paradigms

2. **How do you comment in Python?**
   - Answer: Single line comments use `#`, multi-line comments use triple quotes `'''comment'''` or `"""comment"""`.

3. **What are Python's built-in data types?**
   - Answer: int, float, complex, bool, str, list, tuple, set, dict, None

4. **What is the difference between `==` and `is` in Python?**
   - Answer: `==` compares values, `is` compares object identity (memory address).

5. **How to swap two variables in Python?**
   - Answer: 
     ```python
     a, b = b, a
     ```

6. **What is PEP 8?**
   - Answer: Python Enhancement Proposal that provides coding conventions for Python code.

7. **What are Python's keywords?**
   - Answer: Keywords are reserved words with special meaning (e.g., if, else, for, while, def, class, etc.)

8. **How to check if a variable is a certain type?**
   - Answer: Use `isinstance(var, type)` or `type(var) is type`.

9. **What is the difference between `list` and `tuple`?**
   - Answer: Lists are mutable (can be modified), tuples are immutable. Lists use `[]`, tuples use `()`.

10. **How to reverse a string in Python?**
    - Answer:
      ```python
      s = "hello"
      reversed_s = s[::-1]  # "olleh"
      ```

11. **What is slicing in Python?**
    - Answer: Extracting parts of sequences (strings, lists, tuples) using `[start:stop:step]` notation.

12. **How to concatenate two lists?**
    - Answer:
      ```python
      list1 + list2
      # or
      list1.extend(list2)
      ```

13. **What does the `pass` statement do?**
    - Answer: It's a null operation - nothing happens when it executes. Used as a placeholder.

14. **How to convert a string to lowercase?**
    - Answer:
      ```python
      s = "HELLO"
      s.lower()  # "hello"
      ```

15. **What is the difference between `deep copy` and `shallow copy`?**
    - Answer: Shallow copy creates a new object but doesn't copy nested objects. Deep copy creates a new object and recursively copies all nested objects.

16. **How to generate random numbers in Python?**
    - Answer:
      ```python
      import random
      random.randint(start, end)  # inclusive
      random.random()  # float between 0 and 1
      ```

17. **What is the purpose of `__init__` method?**
    - Answer: It's a constructor method that gets called when an object is instantiated.

18. **How to check if a key exists in a dictionary?**
    - Answer:
      ```python
      if key in my_dict:
      # or
      my_dict.get(key)  # returns None if key doesn't exist
      ```

19. **What is a lambda function?**
    - Answer: An anonymous function defined with `lambda` keyword. Example: `lambda x: x*2`.

20. **How to sort a dictionary by value?**
    - Answer:
      ```python
      sorted_dict = dict(sorted(my_dict.items(), key=lambda item: item[1]))
      ```

21. **What is the difference between `append()` and `extend()` for lists?**
    - Answer: `append()` adds its argument as a single element, while `extend()` iterates over its argument adding each element.

22. **How to remove duplicates from a list?**
    - Answer:
      ```python
      unique_list = list(set(my_list))  # doesn't preserve order
      # or
      from collections import OrderedDict
      unique_list = list(OrderedDict.fromkeys(my_list))  # preserves order
      ```

23. **What is the difference between `remove()`, `pop()` and `del` for lists?**
    - Answer: `remove()` deletes the first matching value, `pop()` removes by index (and returns it), `del` removes by index or slice.

24. **How to check if a string contains a substring?**
    - Answer:
      ```python
      if substring in main_string:
      # or
      main_string.find(substring)  # returns index or -1
      ```

25. **What is the difference between `break` and `continue`?**
    - Answer: `break` exits the loop entirely, `continue` skips to the next iteration.

26. **How to convert a list to a string?**
    - Answer:
      ```python
      ', '.join(['a', 'b', 'c'])  # "a, b, c"
      ```

27. **What is the difference between `*args` and `**kwargs`?**
    - Answer: `*args` collects positional arguments as a tuple, `**kwargs` collects keyword arguments as a dictionary.

28. **How to get the current working directory?**
    - Answer:
      ```python
      import os
      os.getcwd()
      ```

29. **What is the purpose of `if __name__ == "__main__":`?**
    - Answer: Ensures code only runs when the script is executed directly, not when imported as a module.

30. **How to measure execution time of a Python script?**
    - Answer:
      ```python
      import time
      start = time.time()
      # your code
      end = time.time()
      print(end - start)
      ```

---

## <a name="data-structures"></a>Data Structures (30 Questions)

31. **What are the main data structures in Python?**
    - Answer: Lists, tuples, dictionaries, sets, strings, arrays, queues, stacks.

32. **How to create a dictionary with default values?**
    - Answer:
      ```python
      from collections import defaultdict
      d = defaultdict(int)  # default value 0
      ```

33. **What is a namedtuple?**
    - Answer: A factory function from collections module that creates tuple subclasses with named fields.

34. **How to implement a stack in Python?**
    - Answer: Use a list with `append()` for push and `pop()` for pop operations.

35. **How to implement a queue in Python?**
    - Answer:
      ```python
      from collections import deque
      q = deque()
      q.append()  # enqueue
      q.popleft()  # dequeue
      ```

36. **What is the difference between a set and a frozenset?**
    - Answer: Set is mutable, frozenset is immutable.

37. **How to merge two dictionaries?**
    - Answer:
      ```python
      dict1.update(dict2)
      # or Python 3.5+
      merged = {**dict1, **dict2}
      ```

38. **What is a list comprehension?**
    - Answer: A concise way to create lists. Example: `[x**2 for x in range(10)]`.

39. **What is a dictionary comprehension?**
    - Answer: Similar to list comprehension but creates dictionaries. Example: `{x: x**2 for x in range(5)}`.

40. **How to flatten a nested list?**
    - Answer:
      ```python
      flat_list = [item for sublist in nested_list for item in sublist]
      ```

41. **What is the purpose of `collections.Counter`?**
    - Answer: A dict subclass for counting hashable objects.

42. **How to sort a list of dictionaries by a value?**
    - Answer:
      ```python
      sorted_list = sorted(list_of_dicts, key=lambda x: x['key'])
      ```

43. **What is `heapq` module used for?**
    - Answer: Implements heap queue algorithm (priority queue).

44. **How to create a matrix in Python?**
    - Answer:
      ```python
      matrix = [[0 for _ in range(cols)] for _ in range(rows)]
      ```

45. **What is the difference between `list` and `array`?**
    - Answer: Lists can hold different data types, arrays (from array module) hold homogeneous data types.

46. **How to transpose a matrix?**
    - Answer:
      ```python
      transposed = list(zip(*matrix))
      ```

47. **What is `itertools` module used for?**
    - Answer: Provides various functions for creating iterators for efficient looping.

48. **How to find the most common elements in a list?**
    - Answer:
      ```python
      from collections import Counter
      Counter(my_list).most_common(n)
      ```

49. **What is `bisect` module used for?**
    - Answer: Provides support for maintaining lists in sorted order without having to sort after each insertion.

50. **How to create a shallow copy of a list?**
    - Answer:
      ```python
      new_list = old_list.copy()
      # or
      new_list = old_list[:]
      ```

51. **How to create a deep copy of a nested list?**
    - Answer:
      ```python
      import copy
      new_list = copy.deepcopy(old_list)
      ```

52. **What is the difference between `dict.items()`, `dict.keys()`, and `dict.values()`?**
    - Answer: `items()` returns key-value pairs, `keys()` returns keys, `values()` returns values.

53. **How to remove an item from a dictionary?**
    - Answer:
      ```python
      del my_dict[key]
      # or
      my_dict.pop(key)
      ```

54. **What is `OrderedDict`?**
    - Answer: A dictionary subclass that remembers the order entries were added.

55. **How to check if all elements in a list satisfy a condition?**
    - Answer:
      ```python
      all(x > 0 for x in my_list)
      ```

56. **How to check if any element in a list satisfies a condition?**
    - Answer:
      ```python
      any(x > 0 for x in my_list)
      ```

57. **What is the purpose of `enumerate()`?**
    - Answer: Adds a counter to an iterable and returns it as an enumerate object.

58. **What is the purpose of `zip()`?**
    - Answer: Aggregates elements from each of the iterables and returns an iterator of tuples.

59. **How to convert two lists into a dictionary?**
    - Answer:
      ```python
      dict(zip(keys_list, values_list))
      ```

60. **What is `collections.ChainMap`?**
    - Answer: A class for quickly linking multiple mappings so they can be treated as a single unit.


Here are the next sections of the Python question bank with answers:

## <a name="functions"></a>Functions (20 Questions)

61. **How to define a function in Python?**
    - Answer:
      ```python
      def function_name(parameters):
          """docstring"""
          # function body
          return value
      ```

62. **What is the purpose of a docstring?**
    - Answer: A string literal that appears as the first statement in a function, used to document what the function does.

63. **What is the difference between parameters and arguments?**
    - Answer: Parameters are variables in the function definition, arguments are the actual values passed to the function.

64. **What are default arguments?**
    - Answer: Parameters that assume a default value if no argument is provided.
      ```python
      def greet(name="Guest"):
          print(f"Hello {name}")
      ```

65. **What are keyword arguments?**
    - Answer: Arguments passed to a function by explicitly naming each parameter.
      ```python
      greet(name="Alice")
      ```

66. **What are positional arguments?**
    - Answer: Arguments that need to be in the same order as the parameters defined.

67. **What is variable-length argument (*args)?**
    - Answer: Allows a function to accept any number of positional arguments.
      ```python
      def sum_all(*args):
          return sum(args)
      ```

68. **What is keyword variable-length argument (**kwargs)?**
    - Answer: Allows a function to accept any number of keyword arguments.
      ```python
      def print_info(**kwargs):
          for key, value in kwargs.items():
              print(f"{key}: {value}")
      ```

69. **What is a lambda function?**
    - Answer: An anonymous single-expression function.
      ```python
      square = lambda x: x**2
      ```

70. **What is the difference between return and yield?**
    - Answer: `return` terminates the function and returns a value, `yield` produces a generator that can resume execution.

71. **What is a generator function?**
    - Answer: A function that contains `yield` statements and returns a generator iterator.
      ```python
      def count_up_to(n):
          i = 1
          while i <= n:
              yield i
              i += 1
      ```

72. **What is function annotation?**
    - Answer: Metadata about the types used in function parameters and return value.
      ```python
      def greet(name: str) -> str:
          return f"Hello {name}"
      ```

73. **What is recursion in Python?**
    - Answer: When a function calls itself.
      ```python
      def factorial(n):
          return 1 if n == 1 else n * factorial(n-1)
      ```

74. **How to pass a function as an argument?**
    - Answer:
      ```python
      def apply(func, x):
          return func(x)
      ```

75. **What is a closure in Python?**
    - Answer: A nested function that remembers values in the enclosing scope.
      ```python
      def outer(x):
          def inner(y):
              return x + y
          return inner
      ```

76. **What is the purpose of `functools.wraps`?**
    - Answer: Preserves the original function's metadata when creating decorators.

77. **What is a decorator?**
    - Answer: A function that takes another function and extends its behavior.
      ```python
      def my_decorator(func):
          def wrapper():
              print("Before function call")
              func()
              print("After function call")
          return wrapper
      ```

78. **How to use multiple decorators on a single function?**
    - Answer:
      ```python
      @decorator1
      @decorator2
      def my_function():
          pass
      ```

79. **What is `map()` function?**
    - Answer: Applies a function to all items in an input list.
      ```python
      map(lambda x: x**2, [1, 2, 3])
      ```

80. **What is `filter()` function?**
    - Answer: Constructs an iterator from elements of an iterable for which a function returns true.
      ```python
      filter(lambda x: x > 0, [-1, 0, 1])
      ```

## <a name="oop"></a>Object-Oriented Programming (20 Questions)

81. **What are the main principles of OOP?**
    - Answer: Encapsulation, Inheritance, Polymorphism, Abstraction

82. **How to define a class in Python?**
    - Answer:
      ```python
      class MyClass:
          def __init__(self):
              pass
      ```

83. **What is `self` in Python?**
    - Answer: A reference to the current instance of the class, used to access variables and methods.

84. **What is the difference between instance, class, and static methods?**
    - Answer:
      - Instance methods: take `self` parameter, operate on instance data
      - Class methods: take `cls` parameter, marked with `@classmethod`, operate on class
      - Static methods: marked with `@staticmethod`, don't take special parameters

85. **What is inheritance in Python?**
    - Answer: A class can inherit attributes and methods from another class.
      ```python
      class Child(Parent):
          pass
      ```

86. **What is method overriding?**
    - Answer: When a child class provides a different implementation of a method already defined in its parent class.

87. **What is multiple inheritance?**
    - Answer: A class can inherit from multiple parent classes.
      ```python
      class Child(Parent1, Parent2):
          pass
      ```

88. **What is method resolution order (MRO)?**
    - Answer: The order in which Python looks for a method in a hierarchy of classes.

89. **What is polymorphism in Python?**
    - Answer: The ability to use a common interface for multiple forms (data types).

90. **What is encapsulation in Python?**
    - Answer: Restricting access to methods and variables (using private/protected members).

91. **How to make a private variable in Python?**
    - Answer: Prefix with double underscore `__variable`.

92. **What are properties in Python?**
    - Answer: Managed attributes using `@property` decorator.
      ```python
      class MyClass:
          @property
          def value(self):
              return self._value
      ```

93. **What is `__init__` method?**
    - Answer: The constructor method called when an object is instantiated.

94. **What is `__str__` vs `__repr__`?**
    - Answer: `__str__` is for readable representation, `__repr__` is for unambiguous representation.

95. **What is operator overloading?**
    - Answer: Defining special methods like `__add__`, `__sub__` to change how operators work with objects.

96. **What is an abstract class?**
    - Answer: A class that contains one or more abstract methods (from `abc` module).

97. **What is the difference between `isinstance()` and `issubclass()`?**
    - Answer: `isinstance()` checks if an object is an instance of a class, `issubclass()` checks class inheritance.

98. **What are magic methods in Python?**
    - Answer: Special methods with double underscores (e.g., `__len__`, `__iter__`).

99. **What is a mixin class?**
    - Answer: A class that provides methods to other classes but isn't meant to stand alone.

100. **How to implement interfaces in Python?**
     - Answer: Using abstract base classes from the `abc` module.
       ```python
       from abc import ABC, abstractmethod
       class MyInterface(ABC):
           @abstractmethod
           def method(self):
               pass
       ```

## <a name="file-handling"></a>File Handling (15 Questions)

101. **How to open a file in Python?**
     - Answer:
       ```python
       file = open("filename.txt", "mode")  # modes: r, w, a, r+, etc.
       ```

102. **What are the different file modes in Python?**
     - Answer:
       - 'r': read (default)
       - 'w': write (truncates)
       - 'a': append
       - 'r+': read and write
       - 'b': binary mode

103. **How to read an entire file at once?**
     - Answer:
       ```python
       with open("file.txt") as f:
           content = f.read()
       ```

104. **How to read a file line by line?**
     - Answer:
       ```python
       with open("file.txt") as f:
           for line in f:
               print(line)
       ```

105. **How to write to a file?**
     - Answer:
       ```python
       with open("file.txt", "w") as f:
           f.write("content")
       ```

106. **What is the advantage of using `with` statement?**
     - Answer: It automatically closes the file after the block, even if exceptions occur.

107. **How to check if a file exists?**
     - Answer:
       ```python
       import os
       os.path.exists("file.txt")
       ```

108. **How to delete a file?**
     - Answer:
       ```python
       import os
       os.remove("file.txt")
       ```

109. **How to read and write binary files?**
     - Answer:
       ```python
       with open("file.bin", "rb") as f:
           data = f.read()
       with open("file.bin", "wb") as f:
           f.write(data)
       ```

110. **How to read a CSV file?**
     - Answer:
       ```python
       import csv
       with open("file.csv") as f:
           reader = csv.reader(f)
           for row in reader:
               print(row)
       ```

111. **How to write to a CSV file?**
     - Answer:
       ```python
       import csv
       with open("file.csv", "w") as f:
           writer = csv.writer(f)
           writer.writerow(["col1", "col2"])
       ```

112. **How to work with JSON files?**
     - Answer:
       ```python
       import json
       # Reading
       with open("file.json") as f:
           data = json.load(f)
       # Writing
       with open("file.json", "w") as f:
           json.dump(data, f)
       ```

113. **How to handle file paths cross-platform?**
     - Answer: Use `os.path` or `pathlib`:
       ```python
       from pathlib import Path
       Path("folder", "file.txt")
       ```

114. **What is `pickle` module used for?**
     - Answer: Serializing and deserializing Python objects to/from byte streams.

115. **How to read specific lines from a file?**
     - Answer:
       ```python
       with open("file.txt") as f:
           lines = f.readlines()
       line = lines[5]  # 6th line
       ```

## <a name="error-handling"></a>Error Handling (15 Questions)

116. **What are Python's built-in exceptions?**
     - Answer: `BaseException`, `Exception`, `ValueError`, `TypeError`, `IndexError`, `KeyError`, etc.

117. **How to handle exceptions in Python?**
     - Answer:
       ```python
       try:
           # code
       except ExceptionType:
           # handle error
       ```

118. **What is the difference between `try-except` and `try-finally`?**
     - Answer: `except` handles exceptions, `finally` always executes regardless of exceptions.

119. **How to raise an exception?**
     - Answer:
       ```python
       raise ValueError("Invalid value")
       ```

120. **How to create a custom exception?**
     - Answer:
       ```python
       class MyError(Exception):
           pass
       ```

121. **What is the purpose of `else` in try-except?**
     - Answer: Executes if no exceptions occur in the try block.

122. **How to get exception information?**
     - Answer:
       ```python
       try:
           # code
       except Exception as e:
           print(e)  # exception message
           print(type(e))  # exception type
       ```

123. **What is `assert` used for?**
     - Answer: For debugging, raises `AssertionError` if condition is False.
       ```python
       assert x > 0, "x must be positive"
       ```

124. **How to catch multiple exceptions?**
     - Answer:
       ```python
       try:
           # code
       except (TypeError, ValueError) as e:
           # handle
       ```

125. **What is `sys.exc_info()`?**
     - Answer: Returns information about the current exception being handled.

126. **How to ignore an exception?**
     - Answer:
       ```python
       try:
           # code
       except:
           pass
       ```

127. **What is the base class for all exceptions?**
     - Answer: `BaseException`

128. **How to execute cleanup code regardless of exceptions?**
     - Answer: Use `finally` block.

129. **What is `warnings` module used for?**
     - Answer: To issue warning messages without stopping program execution.

130. **How to log exceptions?**
     - Answer:
       ```python
       import logging
       try:
           # code
       except Exception as e:
           logging.exception("Error occurred")
       ```

Here are the remaining sections of the Python question bank:

## <a name="modules-packages"></a>Modules and Packages (15 Questions)

131. **What is a Python module?**
    - Answer: A file containing Python definitions and statements (with `.py` extension) that can be imported.

132. **How to import a module?**
    - Answer:
      ```python
      import module_name
      # or
      from module_name import function_name
      ```

133. **What is `__init__.py` used for?**
    - Answer: Makes a directory a Python package (required in Python 3.3+ for regular packages, optional for namespace packages).

134. **What is the difference between a module and a package?**
    - Answer: A module is a single file, while a package is a directory containing multiple modules and an `__init__.py` file.

135. **How to create a Python package?**
    - Answer:
      1. Create a directory
      2. Add `__init__.py` file
      3. Add module files
      4. Can be installed with `pip` if setup.py is configured

136. **What is `if __name__ == "__main__":` used for?**
    - Answer: Allows code to execute when run directly, but not when imported as a module.

137. **How to reload a module?**
    - Answer:
      ```python
      import importlib
      importlib.reload(module_name)
      ```

138. **What is `sys.path`?**
    - Answer: A list of directories Python searches when importing modules.

139. **How to install third-party packages?**
    - Answer: Using pip: `pip install package_name`

140. **What is `PYTHONPATH`?**
    - Answer: An environment variable specifying additional directories to search for modules.

141. **How to create a namespace package?**
    - Answer: Create directories without `__init__.py` files and ensure they're in Python path.

142. **What is the difference between `import module` and `from module import *`?**
    - Answer: The first imports the whole module (needs `module.function`), the second imports all names directly (can use `function` directly).

143. **How to access a module's documentation?**
    - Answer:
      ```python
      help(module_name)
      # or
      module_name.__doc__
      ```

144. **What are relative imports?**
    - Answer: Importing from the same package using dots:
      ```python
      from .sibling_module import function
      ```

145. **How to list all names in a module?**
    - Answer:
      ```python
      dir(module_name)
      ```

## <a name="standard-library"></a>Python Standard Library (15 Questions)

146. **What are some commonly used standard library modules?**
    - Answer: `os`, `sys`, `math`, `datetime`, `json`, `re`, `collections`, `itertools`, `argparse`, `logging`

147. **How to get current date and time?**
    - Answer:
      ```python
      from datetime import datetime
      now = datetime.now()
      ```

148. **What is `collections.defaultdict`?**
    - Answer: A dictionary subclass that provides default values for missing keys.

149. **How to parse command line arguments?**
    - Answer:
      ```python
      import argparse
      parser = argparse.ArgumentParser()
      parser.add_argument("--name")
      args = parser.parse_args()
      ```

150. **What is `itertools.chain` used for?**
    - Answer: To treat multiple sequences as a single sequence.

151. **How to measure execution time of small code snippets?**
    - Answer:
      ```python
      import timeit
      timeit.timeit('"-".join(str(n) for n in range(100))', number=10000)
      ```

152. **What is `functools.partial` used for?**
    - Answer: To partially apply arguments to a function, creating a new function.

153. **How to implement logging?**
    - Answer:
      ```python
      import logging
      logging.basicConfig(level=logging.INFO)
      logging.info("Message")
      ```

154. **What is `contextlib` used for?**
    - Answer: Utilities for creating and working with context managers.

155. **How to work with temporary files?**
    - Answer:
      ```python
      from tempfile import TemporaryFile
      with TemporaryFile('w+t') as f:
          f.write('Hello')
      ```

156. **What is `unittest` module used for?**
    - Answer: Python's built-in unit testing framework.

157. **How to generate all permutations of a list?**
    - Answer:
      ```python
      from itertools import permutations
      list(permutations([1, 2, 3]))
      ```

158. **What is `glob` module used for?**
    - Answer: Finding pathnames matching a specified pattern.

159. **How to compress files?**
    - Answer:
      ```python
      import gzip
      with gzip.open('file.txt.gz', 'wb') as f:
          f.write(content)
      ```

160. **What is `subprocess` module used for?**
    - Answer: Spawning new processes, connecting to their input/output/error pipes.

## <a name="intermediate-python"></a>Intermediate Python (20 Questions)

161. **What are Python's built-in decorators?**
    - Answer: `@property`, `@classmethod`, `@staticmethod`, `@functools.lru_cache`, etc.

162. **What is the Global Interpreter Lock (GIL)?**
    - Answer: A mutex that protects access to Python objects, preventing multiple threads from executing Python bytecode simultaneously.

163. **How to achieve multithreading in Python?**
    - Answer:
      ```python
      from threading import Thread
      t = Thread(target=function)
      t.start()
      ```

164. **How to achieve multiprocessing in Python?**
    - Answer:
      ```python
      from multiprocessing import Process
      p = Process(target=function)
      p.start()
      ```

165. **What is the difference between multithreading and multiprocessing?**
    - Answer: Threads share memory space, processes have separate memory. GIL affects threads but not processes.

166. **What are Python's memory management techniques?**
    - Answer: Reference counting + garbage collection for cycle detection.

167. **What is a metaclass?**
    - Answer: The class of a class - controls class creation. Default is `type`.

168. **How to implement a singleton pattern?**
    - Answer:
      ```python
      class Singleton:
          _instance = None
          def __new__(cls):
              if cls._instance is None:
                  cls._instance = super().__new__(cls)
              return cls._instance
      ```

169. **What are descriptors?**
    - Answer: Objects that define `__get__`, `__set__`, or `__delete__` methods, used to customize attribute access.

170. **How to implement caching in Python?**
    - Answer:
      ```python
      from functools import lru_cache
      @lru_cache(maxsize=128)
      def expensive_function(x):
          return x * x
      ```

171. **What is monkey patching?**
    - Answer: Modifying or extending code at runtime.

172. **How to make an object callable?**
    - Answer: Implement `__call__` method.

173. **What is the purpose of `__slots__`?**
    - Answer: Optimizes memory usage by restricting instance attributes.

174. **How to implement a context manager?**
    - Answer:
      ```python
      class MyContext:
          def __enter__(self):
              print("Entering")
              return self
          def __exit__(self, exc_type, exc_val, exc_tb):
              print("Exiting")
      ```

175. **What is the purpose of `__del__` method?**
    - Answer: Called when an object is about to be destroyed (not reliable for resource cleanup).

176. **How to implement operator overloading?**
    - Answer: Define special methods like `__add__`, `__sub__`, etc.

177. **What is weakref used for?**
    - Answer: Creating references to objects without preventing garbage collection.

178. **How to implement a custom iterator?**
    - Answer:
      ```python
      class Count:
          def __init__(self, low, high):
              self.current = low
              self.high = high
          def __iter__(self):
              return self
          def __next__(self):
              if self.current > self.high:
                  raise StopIteration
              self.current += 1
              return self.current - 1
      ```

179. **What is the purpose of `__getattr__` and `__getattribute__`?**
    - Answer: `__getattr__` called when attribute not found, `__getattribute__` called for all attribute access.

180. **How to implement a custom container type?**
    - Answer: Implement methods like `__len__`, `__getitem__`, `__setitem__`, `__contains__`.

## <a name="advanced-python"></a>Advanced Python (20 Questions)

181. **What are Python's bytecode and .pyc files?**
    - Answer: Bytecode is the intermediate representation of Python code, .pyc files contain compiled bytecode.

182. **How to profile Python code?**
    - Answer:
      ```python
      import cProfile
      cProfile.run('my_function()')
      ```

183. **What is a coroutine in Python?**
    - Answer: A generalization of subroutines, used for cooperative multitasking with `async/await`.

184. **How to use `asyncio`?**
    - Answer:
      ```python
      import asyncio
      async def main():
          await asyncio.sleep(1)
      asyncio.run(main())
      ```

185. **What is the difference between `yield` and `await`?**
    - Answer: `yield` is for generators, `await` is for coroutines.

186. **How to implement a decorator with arguments?**
    - Answer:
      ```python
      def decorator_with_args(arg):
          def decorator(func):
              def wrapper(*args, **kwargs):
                  print(f"Decorator arg: {arg}")
                  return func(*args, **kwargs)
              return wrapper
          return decorator
      ```

187. **What is a sentinel value?**
    - Answer: A special value used to indicate special conditions (like `None`, `Ellipsis`, or custom objects).

188. **How to implement a custom exception hook?**
    - Answer:
      ```python
      import sys
      def custom_excepthook(type, value, traceback):
          print("Custom exception handling")
      sys.excepthook = custom_excepthook
      ```

189. **What is the purpose of `__subclasses__()`?**
    - Answer: Returns a list of immediate subclasses of a class.

190. **How to implement a plugin architecture?**
    - Answer: Using entry points in setup.py or dynamic imports.

191. **What is the purpose of `inspect` module?**
    - Answer: Get information about live objects like modules, classes, methods, functions, etc.

192. **How to implement a factory pattern?**
    - Answer:
      ```python
      class ShapeFactory:
          def create_shape(self, shape_type):
              if shape_type == "circle":
                  return Circle()
              elif shape_type == "square":
                  return Square()
      ```

193. **What is the purpose of `__mro__` attribute?**
    - Answer: Shows method resolution order for a class.

194. **How to implement a state machine?**
    - Answer: Using classes for states and transitions, or libraries like `transitions`.

195. **What is the purpose of `__debug__`?**
    - Answer: A built-in constant that is `True` if Python was not started with `-O` option.

196. **How to implement memoization?**
    - Answer:
      ```python
      from functools import lru_cache
      @lru_cache(maxsize=None)
      def fib(n):
          return n if n < 2 else fib(n-1) + fib(n-2)
      ```

197. **What is the purpose of `__annotations__`?**
    - Answer: Contains type annotations for variables, class attributes, and function parameters.

198. **How to implement a custom meta class?**
    - Answer:
      ```python
      class Meta(type):
          def __new__(cls, name, bases, dct):
              # customize class creation
              return super().__new__(cls, name, bases, dct)
      class MyClass(metaclass=Meta):
          pass
      ```

199. **What is the purpose of `__getstate__` and `__setstate__`?**
    - Answer: Customize object serialization for `pickle`.

200. **How to implement a descriptor protocol?**
    - Answer:
      ```python
      class Descriptor:
          def __get__(self, obj, objtype=None):
              return obj._value
          def __set__(self, obj, value):
              obj._value = value
      class MyClass:
          attr = Descriptor()
      ```
