# 2. Add Two Numbers 

## Problem Description 

**English**:
You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order, and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list.

**中文**:
給定兩個非空鏈表，表示兩個非負整數。數字以逆序存儲，每個節點包含一個數字。將兩數相加，並將其和作為一個鏈表返回。

## Solution Explanation 

**English**:
To solve this problem, we traverse both linked lists, adding corresponding digits along with any carry from the previous digit addition. We use a dummy head node to simplify edge cases and maintain the resulting linked list.

**中文**:
為了解決這個問題，我們遍歷兩個鏈表，將相應的數字與前一位數相加的進位相加。我們使用一個虛擬頭結點來簡化邊界情況並維護結果鏈表。

## Detailed Steps 

**English**:
1. **Create a Dummy Head**: Create a dummy head node to simplify edge cases.
2. **Initialize Pointers and Carry**: Use a pointer `current` to build the result list and initialize `carry` to 0.
3. **Traverse Both Lists**:
    - Add corresponding digits from both lists along with any carry from the previous addition.
    - Calculate the new digit (`total % 10`) and update the carry (`total // 10`).
    - Create a new node with the calculated digit and link it to the result list.
    - Move pointers `l1` and `l2` to the next nodes if they exist.
4. **Handle Remaining Carry**: If there is any carry left, create a new node with the carry.
5. **Return the Result List**: Return the next node of the dummy head, which is the head of the result list.

**中文**:
1. **創建虛擬頭結點**：創建一個虛擬頭結點來簡化邊界情況。
2. **初始化指針和進位**：使用指針 `current` 來構建結果鏈表，並將 `carry` 初始化為 0。
3. **遍歷兩個鏈表**：
    - 將兩個鏈表中的相應數字與前一次相加的進位相加。
    - 計算新數字（`total % 10`）並更新進位（`total // 10`）。
    - 創建一個包含計算出來的數字的新節點，並將其連接到結果鏈表。
    - 如果 `l1` 和 `l2` 存在，將指針移動到下一個節點。
4. **處理剩餘進位**：如果還有剩餘進位，創建一個包含進位的新節點。
5. **返回結果鏈表**：返回虛擬頭結點的下一個節點，即結果鏈表的頭結點。

## Time Complexity 

**English**:
The time complexity of this solution is **O(max(n, m))**, where `n` and `m` are the lengths of the input linked lists. This is because we traverse both lists once.

**中文**:
該解決方案的時間複雜度是 **O(max(n, m))**，其中 `n` 和 `m` 是輸入鏈表的長度。這是因為我們兩個鏈表各遍歷一次。

## Visual Explanation 

### English:

Suppose we have two linked lists:

- `l1`: 2 -> 4 -> 3 (representing 342)
- `l2`: 5 -> 6 -> 4 (representing 465)

#### Initial State
- `dummy_head` points to a new node with value 0.
- `current` points to `dummy_head`.
- `carry` is initialized to 0.

#### First Iteration
- Add `2` (from `l1`), `5` (from `l2`), and `carry` `0`: `total = 7`.
- Create a new node with value `7` and set `current.next` to this node.
- Move `current` to `current.next`.
- Move `l1` to `4` and `l2` to `6`.

#### Second Iteration
- Add `4` (from `l1`), `6` (from `l2`), and `carry` `0`: `total = 10`.
- Create a new node with value `0` and set `current.next` to this node.
- Set `carry` to `1` (since `10 // 10` is `1`).
- Move `current` to `current.next`.
- Move `l1` to `3` and `l2` to `4`.

#### Third Iteration
- Add `3` (from `l1`), `4` (from `l2`), and `carry` `1`: `total = 8`.
- Create a new node with value `8` and set `current.next` to this node.
- Move `current` to `current.next`.
- Move `l1` and `l2` to `None`.

#### Handle Remaining Carry
- Since `carry` is `0`, no additional node is needed.

Final list: 7 -> 0 -> 8 (representing 807)

### 中文：

假設我們有兩個鏈表：

- `l1`: 2 -> 4 -> 3 (表示 342)
- `l2`: 5 -> 6 -> 4 (表示 465)

#### 初始狀態
- `dummy_head` 指向一個值為 0 的新節點。
- `current` 指向 `dummy_head`。
- `carry` 初始化為 0。

#### 第一次迭代
- 加 `2`（來自 `l1`），`5`（來自 `l2`），和 `carry` `0`：`total = 7`。
- 創建一個值為 `7` 的新節點，並設置 `current.next` 為此節點。
- 將 `current` 移動到 `current.next`。
- 將 `l1` 移動到 `4`，`l2` 移動到 `6`。

#### 第二次迭代
- 加 `4`（來自 `l1`），`6`（來自 `l2`），和 `carry` `0`：`total = 10`。
- 創建一個值為 `0` 的新節點，並設置 `current.next` 為此節點。
- 設置 `carry` 為 `1`（因為 `10 // 10` 是 `1`）。
- 將 `current` 移動到 `current.next`。
- 將 `l1` 移動到 `3`，`l2` 移動到 `4`。

#### 第三次迭代
- 加 `3`（來自 `l1`），`4`（來自 `l2`），和 `carry` `1`：`total = 8`。
- 創建一個值為 `8` 的新節點，並設置 `current.next` 為此節點。
- 將 `current` 移動到 `current.next`。
- 將 `l1` 和 `l2` 移動到 `None`。

#### 處理剩餘進位
- 因為 `carry` 是 `0`，不需要額外的節點。

最終鏈表：7 -> 0 -> 8 (表示 807)

## Code Implementation 

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def addTwoNumbers(self, l1, l2):
    dummy_head = ListNode()
    current = dummy_head
    carry = 0
    while l1 != None or l2 != None or carry != 0:
        val1 = l1.val if l1 is not None else 0
        val2 = l2.val if l2 is not None else 0

        total = val1 + val2 + carry

        # This round actual value is total % 10 because there might be a carry
        val = total % 10
        carry = total // 10
        current.next = ListNode(val)
        current = current.next

        l1 = l1.next if l1 is not None else None
        l2 = l2.next if l2 is not None else None

    return dummy_head.next
