# 20. Valid Parentheses

## Problem Description 

**English**:
Given a string `s` containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.

An input string is valid if:
1. Open brackets must be closed by the same type of brackets.
2. Open brackets must be closed in the correct order.

**中文**:
給定一個只包含字符 '(', ')', '{', '}', '[' 和 ']' 的字符串 `s`，判斷輸入的字符串是否有效。

一個輸入字符串是有效的，如果：
1. 左括號必須由相同類型的右括號閉合。
2. 左括號必須以正確的順序閉合。

## Solution Explanation 

**English**:
To solve this problem, we use a stack to keep track of the opening brackets. As we traverse the string:
1. If we encounter an opening bracket, we push it onto the stack.
2. If we encounter a closing bracket, we check if the stack is empty or if the top of the stack is not the corresponding opening bracket. If either condition is true, the string is invalid.
3. Finally, the string is valid if the stack is empty after processing all characters.

**中文**:
為了解決這個問題，我們使用堆疊來跟踪左括號。當我們遍歷字符串時：
1. 如果遇到左括號，我們將其推入堆疊。
2. 如果遇到右括號，我們檢查堆疊是否為空或堆疊頂部是否不是對應的左括號。如果任一條件為真，則字符串無效。
3. 最後，處理完所有字符後，如果堆疊為空，則字符串有效。

## Detailed Steps 

**English**:
1. **Initialize Data Structures**: Create a dictionary `bracket_map` to map closing brackets to their corresponding opening brackets. Create an empty list `stack` to use as a stack.
2. **Traverse the String**:
    - If the character is an opening bracket, push it onto the stack.
    - If the character is a closing bracket, check if the stack is empty or if the top of the stack does not match the corresponding opening bracket. If either is true, return `False`.
3. **Check Remaining Stack**: After processing all characters, return `True` if the stack is empty, otherwise return `False`.

**中文**:
1. **初始化數據結構**：創建一個字典 `bracket_map`，將右括號映射到對應的左括號。創建一個空列表 `stack` 作為堆疊。
2. **遍歷字符串**：
    - 如果字符是左括號，將其推入堆疊。 
    - 如果字符是右括號，檢查堆疊是否為空或堆疊頂部是否不匹配對應的左括號。如果任一條件為真，則返回 `False`。
      - 堆疊是否為空。如果堆疊為空，說明沒有對應的開括號，返回 `False`。
      - 如果堆疊不為空，彈出堆疊頂部元素，並使用 `bracket_map` 查找當前閉括號對應的開括號，檢查是否匹配。如果不匹配，返回 `False`。
3. **檢查剩餘堆疊**：處理完所有字符後，如果堆疊為空則返回 `True`，否則返回 `False`。

## Time Complexity 

**English**:
The time complexity of this solution is **O(n)**, where `n` is the length of the input string. This is because we traverse the string once and each operation on the stack (push/pop) takes constant time.

**中文**:
該解決方案的時間複雜度是 **O(n)**，其中 `n` 是輸入字符串的長度。這是因為我們遍歷字符串一次，並且每次堆疊操作（推入/彈出）都花費常數時間。

## Visual Explanation 

### English:

Suppose we have a string:

- `s`: "({[]})"

#### Initial State
- `stack` is empty.

#### First Iteration
- `char` is '(': Push onto stack.
- `stack`: ['(']

#### Second Iteration
- `char` is '{': Push onto stack.
- `stack`: ['(', '{']

#### Third Iteration
- `char` is '[': Push onto stack.
- `stack`: ['(', '{', '[']

#### Fourth Iteration
- `char` is ']': Pop from stack (matches '[').
- `stack`: ['(', '{']

#### Fifth Iteration
- `char` is '}': Pop from stack (matches '{').
- `stack`: ['(']

#### Sixth Iteration
- `char` is ')': Pop from stack (matches '(').
- `stack`: []

Final stack is empty, so the string is valid.

### 中文：

假設我們有一個字符串：

- `s`: "({[]})"

#### 初始狀態
- `stack` 是空的。

#### 第一次迭代
- `char` 是 '(': 推入堆疊。
- `stack`: ['(']

#### 第二次迭代
- `char` 是 '{': 推入堆疊。
- `stack`: ['(', '{']

#### 第三次迭代
- `char` 是 '[': 推入堆疊。
- `stack`: ['(', '{', '[']

#### 第四次迭代
- `char` 是 ']': 從堆疊彈出（匹配 '['）。
- `stack`: ['(', '{']

#### 第五次迭代
- `char` 是 '}': 從堆疊彈出（匹配 '{'）。
- `stack`: ['(']

#### 第六次迭代
- `char` 是 ')': 從堆疊彈出（匹配 '('）。
- `stack`: []

最終堆疊為空，因此字符串是有效的。

## Code Implementation 

```python
class Solution:
    def isValid(self, s):
        bracket_map = {')': '(', '}': '{', ']': '['}
        stack = []
        for char in s:
            if char in '([{':
                stack.append(char)
            elif len(stack) == 0 or stack.pop() != bracket_map[char]:
                return False
        return len(stack) == 0

# Example usage:
# s = "({[]})"
# Output: True

solution = Solution()
print(solution.isValid("({[]})"))  # Output: True
print(solution.isValid("({[})"))   # Output: False
