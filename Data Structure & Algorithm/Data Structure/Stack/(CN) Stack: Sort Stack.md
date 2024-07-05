# 堆疊：排序堆疊

## 描述與目標

`sort_stack` 函數接收一個參數，一個堆疊對象。該函數應該使用僅一個額外的堆疊來將堆疊中的元素按升序排序（最低值將位於堆疊的頂部）。

該函數應使用堆疊對象的 `pop`、`push`、`peek` 和 `is_empty` 方法。

**注意**：這是一個新函數，而不是堆疊類中的方法。

## 詳細解釋

要使用另一個堆疊來排序堆疊：

1. **初始化額外的堆疊**：創建一個新的堆疊實例，名為 `sorted_stack`。
2. **排序元素**：當輸入堆疊不為空時，執行以下操作：
   - 從輸入堆疊彈出頂部元素，並將其存儲在變量 `temp` 中。
   - 當 `sorted_stack` 不為空且其頂部元素大於 `temp` 時，從 `sorted_stack` 彈出頂部元素，並將其推回輸入堆疊。
   - 將 `temp` 變量推入 `sorted_stack`。
3. **轉移元素回原始堆疊**：一旦輸入堆疊為空，將元素從 `sorted_stack` 轉移回輸入堆疊。
4. **總結**: 其實就是建立兩個stack，一個當成儲存用的(sorted_stack)，與原始的(stack)，只要遇到小於在原本stack的數字就會全部儲存至sorted_stack，再由小到大排列進去，最小數字先進去最上面是最大的數字。

### 步驟示例：

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

## 程式碼
* **時間複雜度 = O(n^2)**
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
