# Evaluate Reverse Polish Notation 

## Problem Description 

**English**:
Evaluate the value of an arithmetic expression in Reverse Polish Notation (RPN). Valid operators are `+`, `-`, `*`, and `/`. Each operand may be an integer or another expression. Note that division between two integers should truncate toward zero.

**中文**:
計算逆波蘭表示法（RPN）中的算術表達式的值。有效的操作符包括 `+`、`-`、`*` 和 `/`。每個操作數可以是整數或另一個表達式。請注意，兩個整數之間的除法應向零截斷。

## Solution Explanation 

**English**:
To solve this problem, we use a stack to process the tokens of the RPN expression. We traverse each token:
1. If the token is an operand, we convert it to an integer and push it onto the stack.
2. If the token is an operator, we pop the top two elements from the stack, apply the operator, and push the result back onto the stack.
3. At the end, the stack should contain a single element which is the result of the expression.

**中文**:
為了解決這個問題，我們使用堆疊來處理 RPN 表達式的標記。我們遍歷每個標記：
1. 如果標記是操作數，我們將其轉換為整數並推入堆疊。
2. 如果標記是操作符，我們從堆疊中彈出頂部的兩個元素，應用操作符，並將結果推回堆疊。
3. 最後，堆疊中應該只剩下一個元素，這就是表達式的結果。

## Detailed Steps 

**English**:
1. **Initialize Stack**: Create an empty stack.
2. **Traverse Tokens**:
    - If the token is an operand, convert it to an integer and push it onto the stack.
    - If the token is an operator, pop the top two elements from the stack, apply the operator, and push the result back onto the stack.
3. **Return Result**: The final result is the only element left in the stack.

**中文**:
1. **初始化堆疊**：創建一個空堆疊。
2. **遍歷標記**：
    - 如果標記是操作數，將其轉換為整數並推入堆疊。
    - 如果標記是操作符，從堆疊中彈出頂部的兩個元素，應用操作符，並將結果推回堆疊，因為要開始運算。
3. **返回結果**：最終結果是堆疊中唯一剩下的元素。

## Time Complexity

**English**:
The time complexity of this solution is **O(n)**, where `n` is the number of tokens in the RPN expression. This is because each token is processed once.

**中文**:
該解決方案的時間複雜度是 **O(n)**，其中 `n` 是 RPN 表達式中的標記數量。這是因為每個標記只被處理一次。

## Visual Explanation 

### English:

Suppose we have an RPN expression: `["2", "1", "+", "3", "*"]`

#### Initial State
- `stack` is empty.

#### First Token ("2")
- Push 2 onto the stack.
- `stack`: [2]

#### Second Token ("1")
- Push 1 onto the stack.
- `stack`: [2, 1]

#### Third Token ("+")
- Pop 1 and 2 from the stack.
- Calculate 2 + 1 = 3.
- Push 3 onto the stack.
- `stack`: [3]

#### Fourth Token ("3")
- Push 3 onto the stack.
- `stack`: [3, 3]

#### Fifth Token ("*")
- Pop 3 and 3 from the stack.
- Calculate 3 * 3 = 9.
- Push 9 onto the stack.
- `stack`: [9]

Final result: 9

### 中文：

假設我們有一個 RPN 表達式：`["2", "1", "+", "3", "*"]`

#### 初始狀態
- `stack` 是空的。

#### 第一個標記 ("2")
- 將 2 推入堆疊。
- `stack`: [2]

#### 第二個標記 ("1")
- 將 1 推入堆疊。
- `stack`: [2, 1]

#### 第三個標記 ("+")
- 從堆疊中彈出 1 和 2。
- 計算 2 + 1 = 3。
- 將 3 推入堆疊。
- `stack`: [3]

#### 第四個標記 ("3")
- 將 3 推入堆疊。
- `stack`: [3, 3]

#### 第五個標記 ("*")
- 從堆疊中彈出 3 和 3。
- 計算 3 * 3 = 9。
- 將 9 推入堆疊。
- `stack`: [9]

最終結果：9

## Code Implementation 

```python
class Solution:
    def evalRPN(self, tokens):
        stack = []  
        for token in tokens: 
            if token not in "+-*/":  
                stack.append(int(token))  
            else:  
                b = stack.pop()
                a = stack.pop()
                
                if token == '+':
                    result = a + b
                elif token == '-':
                    result = a - b
                elif token == '*':
                    result = a * b
                elif token == '/':
                    result = int(a / b)
                
              
                stack.append(result)
        
        return stack[0]

# Example usage:
sol = Solution()
print(sol.evalRPN(["2", "1", "+", "3", "*"]))  # Output: 9
print(sol.evalRPN(["4", "13", "5", "/", "+"]))  # Output: 6
print(sol.evalRPN(["10", "6", "9", "3", "+", "-11", "*", "/", "*", "17", "+", "5", "+"]))  # Output: 22
