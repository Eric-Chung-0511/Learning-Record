# Stack: Parentheses Balanced 

## Description and Goal

Check to see if a string of parentheses is balanced or not.

By "balanced," we mean that for every open parenthesis, there is a matching closing parenthesis in the correct order. For example, the string "((()))" has three pairs of balanced parentheses, so it is a balanced string. On the other hand, the string "(()))" has an imbalance, as the last two parentheses do not match, so it is not balanced. Also, the string ")(" is not balanced because the close parenthesis needs to follow the open parenthesis.

檢查括號字符串是否平衡。

所謂"平衡"，是指每個開括號都有一個相應的閉括號且順序正確。例如，字符串 "((()))" 有三對平衡的括號，因此是平衡的字符串。而字符串 "(()))" 有不平衡的地方，因為最後兩個括號不匹配，因此不是平衡的。另外，字符串 ")(" 也不是平衡的，因為閉括號需要跟隨開括號。

## Method Name

### is_balanced_parentheses

## Problem Solution Approach

### English:

To check if a string of parentheses is balanced:

1. **Initialize Stack**: Create an empty stack.
2. **Traverse String**: Iterate through each character in the string.
    - **Push**: If the character is an open parenthesis `(`, push it onto the stack. This means we are storing it to find a matching close parenthesis later.
    - **Pop**: If the character is a close parenthesis `)`, check the stack:
        - If the stack is empty, it means there is no corresponding open parenthesis, so return `False`.
        - If the stack is not empty, pop the top element from the stack. This means we have found a matching open parenthesis for the current close parenthesis.
3. **Final Check**: After traversal, check if the stack is empty.
    - If the stack is empty, it means all open parentheses have been matched, so return `True`.(Because they all be pop out)
    - If the stack is not empty, it means there are unmatched open parentheses, so return `False`.

### 中文:

檢查括號字符串是否平衡：

1. **初始化堆疊**：創建一個空堆疊。
2. **遍歷字符串**：迭代字符串中的每個字符。
    - **Push**：如果字符是開括號 `(`，將其推入堆疊。
    - **Pop**：如果字符是閉括號 `)`，檢查堆疊：
        - 如果堆疊為空，代表沒有相應的開括號，因此返回 `False`。
        - 如果堆疊不為空，從堆疊中彈出頂部元素。這代表已經為當前的閉括號找到了匹配的開括號。
3. **最終檢查**：遍歷完成後，檢查堆疊是否為空。
    - 如果堆疊為空，代表所有的開括號都有匹配的閉括號，因此返回 `True`。(因為全都pop out 了)
    - 如果堆疊不為空，代表有未匹配的開括號，因此返回 `False`。


## Code Implementation
* **Time Complexity = O(n)**

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

def is_balanced_parentheses(s):
    stack = Stack()
    for char in s:
        if char == '(':
            stack.push(char)
        elif char == ')':
            if stack.is_empty():
                return False
            stack.pop()
    return stack.is_empty()

# Example usage
print(is_balanced_parentheses("((()))"))  # True
print(is_balanced_parentheses("(()))"))   # False
print(is_balanced_parentheses(")("))      # False
