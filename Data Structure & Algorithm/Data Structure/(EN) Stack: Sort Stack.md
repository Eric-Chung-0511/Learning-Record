# Stack: Sort Stack ( ** Interview Question)

## Description and Goal

The `sort_stack` function takes a single argument, a Stack object. The function should sort the elements in the stack in ascending order (the lowest value will be at the top of the stack) using only one additional stack.

The function should use the `pop`, `push`, `peek`, and `is_empty` methods of the Stack object.

**Note**: This is a new function, not a method within the Stack class.

## Problem Solution Approach

To sort a stack using another stack:

1. **Initialize an Additional Stack**: Create a new instance of the Stack class called `sorted_stack`.
2. **Sort Elements**: While the input stack is not empty, perform the following:
   - Pop the top element from the input stack and store it in a variable `temp`.
   - While the `sorted_stack` is not empty and its top element is greater than `temp`, pop the top element from `sorted_stack` and push it back onto the input stack.
   - Push the `temp` variable onto the `sorted_stack`.
3. **Transfer Elements Back**: Once the input stack is empty, transfer the elements back from `sorted_stack` to the input stack.

### Step-by-Step Example:

Given an initial stack `stack = [5, 1, 3, 2, 4]`, we aim to sort it to become `stack = [1, 2, 3, 4, 5]`.

1. **First Iteration**:
   - Pop `4` from `stack` and set `temp = 4`.
   - `sorted_stack` is empty, so push `temp` onto `sorted_stack`: `sorted_stack = [4]`.

2. **Second Iteration**:
   - Pop `2` from `stack` and set `temp = 2`.
   - `sorted_stack` top `4` is greater than `temp`, so pop `4` from `sorted_stack` and push it back to `stack`: `stack = [5, 1, 3, 4]`.
   - Push `temp` onto `sorted_stack`: `sorted_stack = [2]`.

3. **Third Iteration**:
   - Pop `4` from `stack` and set `temp = 4`.
   - `sorted_stack` top `2` is not greater than `temp`, so push `temp` onto `sorted_stack`: `sorted_stack = [2, 4]`.

4. **Fourth Iteration**:
   - Pop `3` from `stack` and set `temp = 3`.
   - `sorted_stack` top `4` is greater than `temp`, so pop `4` from `sorted_stack` and push it back to `stack`: `stack = [5, 1, 4]`.
   - `sorted_stack` top `2` is not greater than `temp`, so push `temp` onto `sorted_stack`: `sorted_stack = [2, 3]`.

5. **Fifth Iteration**:
   - Pop `4` from `stack` and set `temp = 4`.
   - `sorted_stack` top `3` is not greater than `temp`, so push `temp` onto `sorted_stack`: `sorted_stack = [2, 3, 4]`.

6. **Sixth Iteration**:
   - Pop `1` from `stack` and set `temp = 1`.
   - `sorted_stack` top `4` is greater than `temp`, so pop `4` from `sorted_stack` and push it back to `stack`: `stack = [5, 4]`.
   - `sorted_stack` top `3` is greater than `temp`, so pop `3` from `sorted_stack` and push it back to `stack`: `stack = [5, 4, 3]`.
   - `sorted_stack` top `2` is greater than `temp`, so pop `2` from `sorted_stack` and push it back to `stack`: `stack = [5, 4, 3, 2]`.
   - Push `temp` onto `sorted_stack`: `sorted_stack = [1]`.

7. **Seventh Iteration**:
   - Pop `2` from `stack` and set `temp = 2`.
   - `sorted_stack` top `1` is not greater than `temp`, so push `temp` onto `sorted_stack`: `sorted_stack = [1, 2]`.

8. **Eighth Iteration**:
   - Pop `3` from `stack` and set `temp = 3`.
   - `sorted_stack` top `2` is not greater than `temp`, so push `temp` onto `sorted_stack`: `sorted_stack = [1, 2, 3]`.

9. **Ninth Iteration**:
   - Pop `4` from `stack` and set `temp = 4`.
   - `sorted_stack` top `3` is not greater than `temp`, so push `temp` onto `sorted_stack`: `sorted_stack = [1, 2, 3, 4]`.

10. **Tenth Iteration**:
    - Pop `5` from `stack` and set `temp = 5`.
    - `sorted_stack` top `4` is not greater than `temp`, so push `temp` onto `sorted_stack`: `sorted_stack = [1, 2, 3, 4, 5]`.

11. **Transfer Elements Back**:
    - While `sorted_stack` is not empty, pop elements from `sorted_stack` and push them back to `stack`:
      - `stack = [1, 2, 3, 4, 5]`.

## Code Implementation
* **Time Complexity = O(n^2)**

```python
class Stack:
    def __init__(self):
        self.stack_list = []

    def print_stack(self):
        for i in range(len(self.stack_list) - 1, -1, -1):
            print(self.stack_list[i])

    def is_empty(self):
        return len(self.stack_list) == 0

    def peek(self):
        if self.is_empty():
            return None
        else:
            return self.stack_list[-1]

    def size(self):
        return len(self.stack_list)

    def push(self, value):
        self.stack_list.append(value)

    def pop(self):
        if self.is_empty():
            return None
        else:
            return self.stack_list.pop()

def sort_stack(stack):
    sorted_stack = Stack()
    
    while not stack.is_empty():
        temp = stack.pop()
        
        while not sorted_stack.is_empty() and sorted_stack.peek() > temp:
            stack.push(sorted_stack.pop())
        
        sorted_stack.push(temp)
    
    while not sorted_stack.is_empty():
        stack.push(sorted_stack.pop())

# Example usage
input_stack = Stack()
input_stack.push(5)
input_stack.push(1)
input_stack.push(3)
input_stack.push(2)
input_stack.push(4)

print("Original Stack:")
input_stack.print_stack()

sort_stack(input_stack)

print("Sorted Stack:")
input_stack.print_stack()
