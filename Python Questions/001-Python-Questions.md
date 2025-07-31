### **Python (50 Questions)**  

#### **Lists (10 Questions)**  
1. What is the output of `[1, 2, 3] + [4, 5]`?  
   a) `[1, 2, 3, 4, 5]`  
   b) `[5, 7, 3]`  
   c) `[1, 2, 7, 5]`  
   d) Error  

2. Which method removes the last element from a list?  
   a) `pop()`  
   b) `remove()`  
   c) `delete()`  
   d) `clear()`  

3. How do you reverse a list in Python?  
   a) `list.reverse()`  
   b) `list.sort(reverse=True)`  
   c) `reversed(list)`  
   d) Both a and c  

4. What does `list[1:3]` return if `list = [10, 20, 30, 40]`?  
   a) `[20, 30]`  
   b) `[10, 20]`  
   c) `[20, 30, 40]`  
   d) `[30, 40]`  

5. Which function returns the number of elements in a list?  
   a) `count()`  
   b) `length()`  
   c) `size()`  
   d) `len()`  

6. What is the output of `[x for x in range(5) if x % 2 == 0]`?  
   a) `[0, 1, 2, 3, 4]`  
   b) `[0, 2, 4]`  
   c) `[1, 3]`  
   d) `[2, 4]`  

7. How do you check if an element exists in a list?  
   a) `if element in list`  
   b) `list.exists(element)`  
   c) `list.contains(element)`  
   d) `if list.has(element)`  

8. What does `list.append([4,5])` do?  
   a) Adds `4` and `5` as separate elements  
   b) Adds `[4,5]` as a nested list  
   c) Raises an error  
   d) Replaces the last element  

9. What is the output of `[1, 2] * 2`?  
   a) `[2, 4]`  
   b) `[1, 2, 1, 2]`  
   c) `[1, 2, 2]`  
   d) Error  

10. Which method removes the first occurrence of a value in a list?  
    a) `pop()`  
    b) `remove()`  
    c) `delete()`  
    d) `discard()`  

#### **Tuples (5 Questions)**  
11. Tuples are:  
    a) Mutable  
    b) Immutable  
    c) Dynamic  
    d) None  

12. What is the output of `tuple = (1,); type(tuple)`?  
    a) `<class 'int'>`  
    b) `<class 'tuple'>`  
    c) `<class 'list'>`  
    d) Error  

13. How do you convert a list to a tuple?  
    a) `tuple(list)`  
    b) `list.to_tuple()`  
    c) `tuple.from_list(list)`  
    d) `tuple = list`  

14. What is the output of `(1, 2) + (3, 4)`?  
    a) `(1, 2, 3, 4)`  
    b) `(4, 6)`  
    c) `(1, 2, 7)`  
    d) Error  

15. Which is faster for fixed data?  
    a) List  
    b) Tuple  
    c) Both are equal  
    d) Depends on data  

#### **Dictionaries (10 Questions)**  
16. How do you access a value in a dictionary?  
    a) `dict.get(key)`  
    b) `dict[key]`  
    c) Both a and b  
    d) `dict.value(key)`  

17. What is the output of `dict = {1: 'a', 2: 'b'}; dict.pop(1)`?  
    a) `'a'`  
    b) `{2: 'b'}`  
    c) `1`  
    d) `{1: 'a'}`  

18. Which method returns all keys in a dictionary?  
    a) `dict.keys()`  
    b) `dict.get_keys()`  
    c) `dict.all_keys()`  
    d) `dict.key_list()`  

19. What does `dict.update({'x': 10})` do?  
    a) Adds `{'x': 10}` if key doesn’t exist  
    b) Updates value if key exists  
    c) Both a and b  
    d) Deletes the key  

20. What is the output of `{x: x**2 for x in range(3)}`?  
    a) `{0: 0, 1: 1, 2: 4}`  
    b) `[0, 1, 4]`  
    c) `{1, 4, 9}`  
    d) Error  

21. How do you check if a key exists in a dictionary?  
    a) `if key in dict`  
    b) `dict.has_key(key)`  
    c) `dict.exists(key)`  
    d) `if dict.contains(key)`  

22. What is the output of `dict = {}; dict[1] = 'a'; dict[1] = 'b'; dict[1]`?  
    a) `'a'`  
    b) `'b'`  
    c) `KeyError`  
    d) `None`  

23. Which method removes all items from a dictionary?  
    a) `dict.clear()`  
    b) `dict.remove_all()`  
    c) `dict.delete()`  
    d) `dict.empty()`  

24. What does `dict.items()` return?  
    a) List of tuples (key, value)  
    b) List of keys  
    c) List of values  
    d) None  

25. How do you merge two dictionaries?  
    a) `dict1 + dict2`  
    b) `dict1.update(dict2)`  
    c) `merge(dict1, dict2)`  
    d) `dict1.join(dict2)`  

#### **Functions (10 Questions)**  
26. What is the output of `def f(x=1): return x*2; print(f(3))`?  
    a) `6`  
    b) `1`  
    c) `3`  
    d) Error  

27. What does `*args` represent in a function?  
    a) Keyword arguments  
    b) Variable-length non-keyword args  
    c) Mandatory arguments  
    d) Return value  

28. What is the output of `lambda x: x+2` with `x=3`?  
    a) `5`  
    b) `3`  
    c) `lambda 5`  
    d) Error  

29. Which keyword is used to return a value?  
    a) `break`  
    b) `return`  
    c) `yield`  
    d) `exit`  

30. What is a generator function?  
    a) Uses `return`  
    b) Uses `yield`  
    c) Returns a list  
    d) None  

31. What is recursion?  
    a) A function calling itself  
    b) A loop  
    c) A built-in function  
    d) A decorator  

32. What is the scope of a variable defined inside a function?  
    a) Global  
    b) Local  
    c) Both  
    d) None  

33. What does `map(lambda x: x*2, [1,2,3])` return?  
    a) `[2,4,6]`  
    b) `<map object>`  
    c) `(2,4,6)`  
    d) Error  

34. What is a decorator?  
    a) A function modifying another function  
    b) A design pattern  
    c) A built-in function  
    d) A class  

35. What is the output of `print(*(1,2,3))`?  
    a) `(1,2,3)`  
    b) `1 2 3`  
    c) `6`  
    d) Error  

#### **OOP (15 Questions)**  
36. What is encapsulation?  
    a) Hiding implementation details  
    b) Inheriting properties  
    c) Bundling data and methods  
    d) Both a and c  

37. What is inheritance?  
    a) Deriving a class from another  
    b) Overriding methods  
    c) Both a and b  
    d) None  

38. What is polymorphism?  
    a) Same method name, different actions  
    b) Method overriding  
    c) Both a and b  
    d) None  

39. What is `__init__`?  
    a) Constructor  
    b) Destructor  
    c) Class method  
    d) Static method  

40. What is the output of:  
    ```python
    class A: pass  
    print(A())  
    ```  
    a) `<__main__.A object at 0x...>`  
    b) `A`  
    c) `None`  
    d) Error  

41. How do you make a method static?  
    a) `@staticmethod`  
    b) `@classmethod`  
    c) `static()`  
    d) `@static`  

42. What is method overriding?  
    a) Redefining a parent method in child  
    b) Overloading a method  
    c) Both  
    d) None  

43. What is `super()` used for?  
    a) Access parent class methods  
    b) Call child methods  
    c) Both  
    d) None  

44. What is the output of:  
    ```python
    class A:  
        def __init__(self): print("A")  
    class B(A):  
        def __init__(self): print("B")  
    B()  
    ```  
    a) `A`  
    b) `B`  
    c) `A B`  
    d) Error  

45. What is `self` in Python?  
    a) Reference to the instance  
    b) Keyword  
    c) Both  
    d) None  

46. What is an abstract class?  
    a) Cannot be instantiated  
    b) Contains abstract methods  
    c) Both  
    d) None  

47. What is the output of:  
    ```python
    class A:  
        @classmethod  
        def f(cls): print(cls.__name__)  
    A.f()  
    ```  
    a) `A`  
    b) `cls`  
    c) `f`  
    d) Error  

48. What is operator overloading?  
    a) Defining `__add__` for `+`  
    b) Changing operator behavior  
    c) Both  
    d) None  

49. What is `__str__` used for?  
    a) String representation  
    b) Object initialization  
    c) Both  
    d) None  

50. What is multiple inheritance?  
    a) Inheriting from multiple classes  
    b) Overriding multiple methods  
    c) Both  
    d) None  

---

### **Excel (15 Questions)**  

51. Which function adds numbers in a range?  
    a) `SUM()`  
    b) `ADD()`  
    c) `TOTAL()`  
    d) `COUNT()`  

52. What does `VLOOKUP` do?  
    a) Searches vertically  
    b) Returns a value from a table  
    c) Both  
    d) None  

53. What is `$A$1` called?  
    a) Relative reference  
    b) Absolute reference  
    c) Mixed reference  
    d) None  

54. Which shortcut opens the Format Cells dialog?  
    a) `Ctrl+1`  
    b) `Ctrl+F`  
    c) `Alt+F1`  
    d) `Shift+F1`  

55. What does `COUNTIF` do?  
    a) Counts cells meeting a condition  
    b) Sums cells  
    c) Both  
    d) None  

56. Which chart is best for trends?  
    a) Pie  
    b) Line  
    c) Bar  
    d) Scatter  

57. What does `IFERROR` do?  
    a) Checks for errors  
    b) Returns alternative if error  
    c) Both  
    d) None  

58. What is `CONCATENATE` used for?  
    a) Joining text  
    b) Splitting text  
    c) Both  
    d) None  

59. Which function returns today’s date?  
    a) `TODAY()`  
    b) `NOW()`  
    c) `DATE()`  
    d) `TIME()`  

60. What is a PivotTable used for?  
    a) Data summarization  
    b) Data analysis  
    c) Both  
    d) None  

61. What does `INDEX-MATCH` do?  
    a) Alternative to `VLOOKUP`  
    b) Faster lookup  
    c) Both  
    d) None  

62. What is `TRIM` used for?  
    a) Remove extra spaces  
    b) Split text  
    c) Both  
    d) None  

63. What does `F4` do in Excel?  
    a) Toggle absolute/relative ref  
    b) Repeat last action  
    c) Both  
    d) None  

64. Which function checks if a condition is met?  
    a) `IF()`  
    b) `CHECK()`  
    c) `TEST()`  
    d) `CONDITION()`  

65. What is `Data Validation` used for?  
    a) Restrict input  
    b) Validate data  
    c) Both  
    d) None  

---

### **PowerBI (15 Questions)**  

66. What is DAX?  
    a) Data Analysis Expressions  
    b) PowerBI query language  
    c) Both  
    d) None  

67. What is a measure in PowerBI?  
    a) Dynamic calculation  
    b) Column  
    c) Both  
    d) None  

68. What is Power Query used for?  
    a) Data transformation  
    b) Data modeling  
    c) Both  
    d) None  

69. What is a calculated column?  
    a) Static column  
    b) Computed at refresh  
    c) Both  
    d) None  

70. What is the main visualization in PowerBI?  
    a) Charts  
    b) Tables  
    c) Both  
    d) None  

71. What does `RELATIONSHIP` do?  
    a) Links tables  
    b) Filters data  
    c) Both  
    d) None  

72. What is `CALCULATE` used for?  
    a) Modify filter context  
    b) Aggregate data  
    c) Both  
    d) None  

73. What is a slicer?  
    a) Interactive filter  
    b) Visual  
    c) Both  
    d) None  

74. What is `SUMMARIZE` used for?  
    a) Group data  
    b) Create tables  
    c) Both  
    d) None  

75. What is PowerBI Desktop?  
    a) Authoring tool  
    b) Cloud service  
    c) Both  
    d) None  

76. What is `DISTINCTCOUNT` used for?  
    a) Count unique values  
    b) Sum values  
    c) Both  
    d) None  

77. What is `M` language?  
    a) Power Query language  
    b) DAX alternative  
    c) Both  
    d) None  

78. What is a drill-through filter?  
    a) Navigates to detailed data  
    b) Filters reports  
    c) Both  
    d) None  

79. What is `Q&A` in PowerBI?  
    a) Natural language query  
    b) AI feature  
    c) Both  
    d) None  

80. What is `PowerBI Service`?  
    a) Cloud platform  
    b) Report sharing  
    c) Both  
    d) None  



**8 WAP (Write a Program)** questions covering **Python** concepts:  

---

### **Functions (4 Questions)**  

1. **Factorial Calculator**  
   Write a Python function `factorial(n)` that calculates the factorial of a number `n` using recursion.  
   **Input:** `5`  
   **Output:** `120`  

2. **Prime Number Checker**  
   Write a function `is_prime(num)` that returns `True` if the number is prime, otherwise `False`.  
   **Input:** `13`  
   **Output:** `True`  

3. **List Operations**  
   Write a function `list_operations(lst)` that takes a list and returns:  
   - Sum of all elements  
   - Maximum value  
   - Reversed list  
   **Input:** `[1, 2, 3, 4]`  
   **Output:** `(10, 4, [4, 3, 2, 1])`  

4. **Lambda & Map**  
   Write a program using `lambda` and `map` to square each element in a list.  
   **Input:** `[1, 2, 3, 4]`  
   **Output:** `[1, 4, 9, 16]`  

---

### **OOPs (4 Questions)**  

5. **Class - Rectangle Area**  
   Create a class `Rectangle` with methods to:  
   - Initialize `length` and `width`  
   - Calculate `area()`  
   - Calculate `perimeter()`  
   **Input:**  
   ```python
   r = Rectangle(4, 5)  
   print(r.area())  
   ```  
   **Output:** `20`  

6. **Inheritance - Animal Sounds**  
   Create a base class `Animal` with a method `sound()` and derive `Dog` and `Cat` classes overriding `sound()`.  
   **Input:**  
   ```python
   d = Dog()  
   print(d.sound())  
   ```  
   **Output:** `Bark!`  

7. **Encapsulation - Bank Account**  
   Create a class `BankAccount` with private attributes `__balance` and methods to `deposit()`, `withdraw()`, and `get_balance()`.  
   **Input:**  
   ```python
   acc = BankAccount(1000)  
   acc.withdraw(200)  
   print(acc.get_balance())  
   ```  
   **Output:** `800`  

8. **Polymorphism - Shape Area**  
   Create classes `Circle` and `Square` with a method `area()`. Use polymorphism to compute areas.  
   **Input:**  
   ```python
   shapes = [Circle(3), Square(4)]  
   for shape in shapes:  
       print(shape.area())  
   ```  
   **Output:**  
   ```  
   28.274  
   16  
   ```  