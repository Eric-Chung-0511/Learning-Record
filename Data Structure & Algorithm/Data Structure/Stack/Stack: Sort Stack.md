# Stack: Sort Stack 

## Description and Goal

**English:**
The `sort_stack` function takes a single argument, a Stack object. The function should sort the elements in the stack in ascending order (the lowest value will be at the top of the stack) using only one additional stack.

The function should use the `pop`, `push`, `peek`, and `is_empty` methods of the Stack object.

**中文:**
`sort_stack` 函數接收一個參數，一個堆疊對象。該函數應該使用僅一個額外的堆疊來將堆疊中的元素按升序排序（最低值將位於堆疊的頂部）。

該函數應使用堆疊對象的 `pop`、`push`、`peek` 和 `is_empty` 方法。

**Note**: This is a new function, not a method within the Stack class.

## Problem Solution Approach

### English

To sort a stack using another stack:

1. **Initialize an Additional Stack**: Create a new instance of the Stack class called `sorted_stack`.
2. **Sort Elements**: While the input stack is not empty, perform the following:
   - Pop the top element from the input stack and store it in a variable `temp`.
   - While the `sorted_stack` is not empty and its top element is greater than `temp`, pop the top element from `sorted_stack` and push it back onto the input stack.
   - Push the `temp` variable onto the `sorted_stack`.
3. **Transfer Elements Back**: Once the input stack is empty, transfer the elements back from `sorted_stack` to the input stack.

### 中文

1. **初始化額外的堆疊**：創建一個新的堆疊實例，名為 `sorted_stack`。
2. **排序元素**：當輸入堆疊不為空時，執行以下操作：
   - 從輸入堆疊彈出頂部元素，並將其存儲在變量 `temp` 中。
   - 當 `sorted_stack` 不為空且其頂部元素大於 `temp` 時，從 `sorted_stack` 彈出頂部元素，並將其推回輸入堆疊。
   - 將 `temp` 變量推入 `sorted_stack`。
3. **轉移元素回原始堆疊**：一旦輸入堆疊為空，將元素從 `sorted_stack` 轉移回輸入堆疊。
4. **總結**: 其實就是建立兩個stack，一個當成儲存用的(sorted_stack)，與原始的(stack)，只要遇到小於在原本stack的數字就會全部儲存至sorted_stack，再由小到大排列進去，最小數字先進去最上面是最大的數字。

## Step-by-Step Example:

#### English

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
     
12. **Conclusion**:
    1. First, we take a number temp from the stack.
    
    2. Then, we compare temp with the top number of the sorted_stack. If the sorted_stack is empty, or if temp is greater than or equal to the top number of the sorted_stack, we put temp into the              sorted_stack.

    3. If temp is smaller than the top number of the sorted_stack, we take the top number of the sorted_stack and put it back into the stack, until we find the appropriate position to place temp.

    4. Repeat this process until the stack is empty.

    5. Finally, the numbers in the sorted_stack are arranged in ascending order. We then put them back into the stack one by one, so that the numbers in the stack are also arranged in ascending order.
   
#### 中文

給定一個初始堆疊 `stack = [5, 1, 3, 2, 4]`，我們希望將其排序為 `stack = [1, 2, 3, 4, 5]`。

1. **第一次迭代**：
   - 從 `stack` 彈出 `4` 並設置 `temp = 4`。
   - `sorted_stack` 為空，將 `temp` 推入 `sorted_stack`：`sorted_stack = [4]`。

2. **第二次迭代**：
   - 從 `stack` 彈出 `2` 並設置 `temp = 2`。
   - `sorted_stack` 的頂部元素 `4` 大於 `temp`，所以將 `4` 從 `sorted_stack` 彈出並推回 `stack`：`stack = [5, 1, 3, 4]`。
   - 將 `temp` 推入 `sorted_stack`：`sorted_stack = [2]`。

3. **第三次迭代**：
   - 從 `stack` 彈出 `4` 並設置 `temp = 4`。
   - `sorted_stack` 的頂部元素 `2` 不大於 `temp`，所以將 `temp` 推入 `sorted_stack`：`sorted_stack = [2, 4]`。

4. **第四次迭代**：
   - 從 `stack` 彈出 `3` 並設置 `temp = 3`。
   - `sorted_stack` 的頂部元素 `4` 大於 `temp`，所以將 `4` 從 `sorted_stack` 彈出並推回 `stack`：`stack = [5, 1, 4]`。
   - `sorted_stack` 的頂部元素 `2` 不大於 `temp`，所以將 `temp` 推入 `sorted_stack`：`sorted_stack = [2, 3]`。

5. **第五次迭代**：
   - 從 `stack` 彈出 `4` 並設置 `temp = 4`。
   - `sorted_stack` 的頂部元素 `3` 不大於 `temp`，所以將 `temp` 推入 `sorted_stack`：`sorted_stack = [2, 3, 4]`。

6. **第六次迭代**：
   - 從 `stack` 彈出 `1` 並設置 `temp = 1`。
   - `sorted_stack` 的頂部元素 `4` 大於 `temp`，所以將 `4` 從 `sorted_stack` 彈出並推回 `stack`：`stack = [5, 4]`。
   - `sorted_stack` 的頂部元素 `3` 大於 `temp`，所以將 `3` 從 `sorted_stack` 彈出並推回 `stack`：`stack = [5, 4, 3]`。
   - `sorted_stack` 的頂部元素 `2` 大於 `temp`，所以將 `2` 從 `sorted_stack` 彈出並推回 `stack`：`stack = [5, 4, 3, 2]`。
   - 將 `temp` 推入 `sorted_stack`：`sorted_stack = [1]`。

7. **第七次迭代**：
   - 從 `stack` 彈出 `2` 並設置 `temp = 2`。
   - `sorted_stack` 的頂部元素 `1` 不大於 `temp`，所以將 `temp` 推入 `sorted_stack`：`sorted_stack = [1, 2]`。

8. **第八次迭代**：
   - 從 `stack` 彈出 `3` 並設置 `temp = 3`。
   - `sorted_stack` 的頂部元素 `2` 不大於 `temp`，所以將 `temp` 推入 `sorted_stack`：`sorted_stack = [1, 2, 3]`。

9. **第九次迭代**：
   - 從 `stack` 彈出 `4` 並設置 `temp = 4`。
   - `sorted_stack` 的頂部元素 `3` 不大於 `temp`，所以將 `temp` 推入 `sorted_stack`：`sorted_stack = [1, 2, 3, 4]`。

10. **第十次迭代**：
    - 從 `stack` 彈出 `5` 並設置 `temp = 5`。
    - `sorted_stack` 的頂部元素 `4` 不大於 `temp`，所以將 `temp` 推入 `sorted_stack`：`sorted_stack = [1, 2, 3, 4, 5]`。

11. **轉移元素回原始堆疊**：
    - 當 `sorted_stack` 不為空時，從 `sorted_stack` 彈出元素並推入 `stack`：
      - `stack = [1, 2, 3, 4, 5]`。

12. **總結**:
    1. 首先，我們從 `stack` 中取出一個數字 `temp`。

    2. 然後，我們比較 `temp` 和 `sorted_stack` 頂部的數字。如果 `sorted_stack` 是空的，或者 `temp` 比 `sorted_stack` 頂部的數字大或相等，就把 `temp` 放到 `sorted_stack` 中。

    3. 如果 `temp` 比 `sorted_stack` 頂部的數字小，我們就把 `sorted_stack` 頂部的數字拿出來，放回 `stack`，直到找到合適的位置把 `temp` 放進去。

    4. 重複這個過程，直到 `stack` 變空。

    5. 最後，`sorted_stack` 中的數字就是按從小到大的順序排列的，我們再把它們一個個放回 `stack` 中，這樣 `stack` 中的數字也就按從小到大的順序排列了。


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
