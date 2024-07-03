# Stack: Reverse String 

## Description and Goal

The `reverse_string` function takes a single parameter, `string`, which is the string you want to reverse. The function should return a new string with the letters in reverse order. This will use the `Stack` class we created in the previous coding exercises.

**Note**: This is a new function, not a method within the Stack class.

## Problem Solution Approach

### English:

To reverse a string using a stack:

1. **Initialize a Stack**: Create an empty stack to store the characters of the string.
2. **Push Characters onto Stack**: Traverse the string and push each character onto the stack.
3. **Pop Characters from Stack**: Pop characters from the stack and append them to a new string. Since the stack is LIFO (Last In, First Out), the characters will be in reverse order.
4. **Return the Reversed String**: Return the new string with characters in reverse order.

### Step-by-Step Example:

Given an initial string `string = "hello"`, we aim to reverse it to become `"olleh"`.

1. **Initialize Stack**:
   - `stack = []`

2. **Push Characters onto Stack**:
   - Push `h` onto `stack`: `stack = ['h']`
   - Push `e` onto `stack`: `stack = ['h', 'e']`
   - Push `l` onto `stack`: `stack = ['h', 'e', 'l']`
   - Push `l` onto `stack`: `stack = ['h', 'e', 'l', 'l']`
   - Push `o` onto `stack`: `stack = ['h', 'e', 'l', 'l', 'o']`

3. **Pop Characters from Stack**:
   - Pop `o` from `stack` and append to `reversed_string`: `reversed_string = 'o'`
   - Pop `l` from `stack` and append to `reversed_string`: `reversed_string = 'ol'`
   - Pop `l` from `stack` and append to `reversed_string`: `reversed_string = 'oll'`
   - Pop `e` from `stack` and append to `reversed_string`: `reversed_string = 'olle'`
   - Pop `h` from `stack` and append to `reversed_string`: `reversed_string = 'olleh'`

4. **Return the Reversed String**:
   - Return `reversed_string = "olleh"`

### 中文：

要使用堆疊來反轉字符串：

1. **初始化堆疊**：創建一個空堆疊，用於存儲字符串的字符。
2. **將字符推入堆疊**：遍歷字符串並將每個字符推入堆疊。
3. **從堆疊中彈出字符**：從堆疊中彈出字符並將它們附加到新字符串中。由於堆疊是後進先出（LIFO），字符將以相反的順序排列。
4. **返回反轉後的字符串**：返回包含字符的反轉順序的新字符串。

### 步驟示例：

給定初始字符串 `string = "hello"`，將其反轉為 `"olleh"`。

1. **初始化堆疊**：
   - `stack = []`

2. **將字符推入堆疊**：
   - 將 `h` 推入 `stack`：`stack = ['h']`
   - 將 `e` 推入 `stack`：`stack = ['h', 'e']`
   - 將 `l` 推入 `stack`：`stack = ['h', 'e', 'l']`
   - 將 `l` 推入 `stack`：`stack = ['h', 'e', 'l', 'l']`
   - 將 `o` 推入 `stack`：`stack = ['h', 'e', 'l', 'l', 'o']`

3. **從堆疊中彈出字符**：
   - 從 `stack` 中彈出 `o` 並附加到 `reversed_string`：`reversed_string = 'o'`
   - 從 `stack` 中彈出 `l` 並附加到 `reversed_string`：`reversed_string = 'ol'`
   - 從 `stack` 中彈出 `l` 並附加到 `reversed_string`：`reversed_string = 'oll'`
   - 從 `stack` 中彈出 `e` 並附加到 `reversed_string`：`reversed_string = 'olle'`
   - 從 `stack` 中彈出 `h` 並附加到 `reversed_string`：`reversed_string = 'olleh'`

4. **返回反轉後的字符串**：
   - 返回 `reversed_string = "olleh"`

## Code Implementation

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

def reverse_string(string):
    stack = Stack()
    
    # Push all characters of the string onto the stack
    for char in string:
        stack.push(char)
    
    # Initialize an empty string to collect the reversed characters
    reversed_string = ''
    
    # Pop characters from the stack to form the reversed string
    while not stack.is_empty():
        reversed_string += stack.pop()
    
    return reversed_string

# Example usage
print(reverse_string("hello"))  # Output: "olleh"
print(reverse_string("Stack"))  # Output: "kcatS"
