# Leetcode 21. Merge Two Sorted Lists 

## Problem Description

**English**:
Given the heads of two sorted linked lists `list1` and `list2`, merge the two lists into one sorted list. The list should be made by splicing together the nodes of the first two lists. Return the head of the merged linked list.

**中文**:
給定兩個排序鏈表的頭結點 `list1` 和 `list2`，將這兩個鏈表合併成一個排序鏈表。該鏈表應由前兩個鏈表的節點拼接而成。返回合併後的鏈表的頭結點。

## Solution Explanation 

**English**:
To solve this problem, we need to merge the two sorted linked lists into a new sorted linked list. We can do this by using an additional linked list to store the result, comparing the nodes of the two lists one by one, and adding the smaller node to the result list.

**中文**:
為了解決這個問題，我們需要將兩個排序鏈表合併成一個新的排序鏈表。我們可以通過使用額外的鏈表來存儲結果，一次比較兩個鏈表的節點，並將較小的節點添加到結果鏈表中。

## Detailed Steps

**English**:
1. **Create a Dummy Node**: Create a dummy node `head` which helps in easily handling edge cases.
2. **Pointer to Track Current Node**: Use a pointer `current` to keep track of the current node in the result list.
3. **Traverse List1 and List2**:
    - Compare the current nodes of both lists.
    - Append the smaller node to the result list.
    - Move the pointer of the list from which the node was taken.
4. **Append Remaining Nodes**: After the loop, if one list is exhausted(None), append the remaining part of the other list to the result list.
5. **Return the Merged List**: Return the next node of `head`, which is the head of the merged list.

Suppose we have two sorted linked lists:

- `list1`: 1 -> 3 -> 5
- `list2`: 2 -> 4 -> 6

#### Initial State
- `head` and `current` both point to a new dummy head node with `val` 0.

#### First Iteration
- `list1`'s current node value 1 is less than `list2`'s current node value 2, so `current.next = list1`.
- Move `list1` to the next node, now `list1` is 3 -> 5.
- Move `current` to the next node, now `current` is 1.

#### Second Iteration
- `list1`'s current node value 3 is greater than `list2`'s current node value 2, so `current.next = list2`.
- Move `list2` to the next node, now `list2` is 4 -> 6.
- Move `current` to the next node, now `current` is 2.

Continue this process until one list is exhausted. Suppose `list2` is exhausted first, and the remaining nodes are in `list1`:

- `current.next = list1` sets the `current.next` pointer to `list1`.

The final result is 1 -> 2 -> 3 -> 4 -> 5 -> 6.

Through these steps, the pointers `current` and `head` gradually link all nodes together, forming the final merged list.


**中文**:
1. **創建一個虛擬節點**：創建一個虛擬節點 `head`，這有助於輕鬆處理邊界情況。
2. **指針跟踪當前節點**：使用指針 `current` 來跟踪結果鏈表中的當前節點。
3. **遍歷 List1 和 List2**：
    - 比較兩個鏈表的當前節點。
    - 將較小的節點附加到結果鏈表。
    - 移動節點所取鏈表的指針。
4. **附加剩餘節點**：循環結束後，如果一個鏈表耗盡(None)，將另一個鏈表的剩餘部分附加到結果鏈表。
5. **返回合併後的鏈表**：返回 `head` 的下一個節點，這是合併後的鏈表的頭結點。

假設有兩個已排序的鏈表：

- `list1`: 1 -> 3 -> 5
- `list2`: 2 -> 4 -> 6

#### 初始狀態
- `head` 和 `current` 都指向一個新的虛擬頭節點，其 `val` 為 0。

#### 第一次迴圈
- `list1` 的當前節點值 1 小於 `list2` 的當前節點值 2，所以 `current.next = list1`。
- `list1` 移動到下一個節點，現在 `list1` 為 3 -> 5。
- `current` 移動到下一個節點，現在 `current` 為 1。

#### 第二次迴圈
- `list1` 的當前節點值 3 大於 `list2` 的當前節點值 2，所以 `current.next = list2`。
- `list2` 移動到下一個節點，現在 `list2` 為 4 -> 6。
- `current` 移動到下一個節點，現在 `current` 為 2。

依此類推，直到其中一個鏈表遍歷完。假設 `list2` 先遍歷完，剩下的節點在 `list1`：

- `current.next = list1` 將 `current.next` 指針設置為 `list1`。

最終結果為 1 -> 2 -> 3 -> 4 -> 5 -> 6。

透過上述步驟，`current` 和 `head` 的指針逐漸連接所有節點，形成最終的合併鏈表。

## Time Complexity

**English**:
The time complexity of this solution is **O(n + m)**, where `n` and `m` are the lengths of `list1` and `list2`, respectively. This is because we traverse each list once.

**中文**:
該答案的時間複雜度是 **O(n + m)**，其中 `n` 和 `m` 分別是 `list1` 和 `list2` 的長度。這是因為我們每個鏈表只遍歷一次。

## Code Implementation 

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def mergeTwolists(list1, list2):
    head = ListNode()
    current = head
    
    while list1 and list2:
        if list1.val <= list2.val:
            current.next = list1
            list1 = list1.next
        else:
            current.next = list2
            list2 = list2.next
        current = current.next
    
    # One of list1 and list2 can still have remaining nodes, attach them
    current.next = list1 or list2
    
    return head.next

