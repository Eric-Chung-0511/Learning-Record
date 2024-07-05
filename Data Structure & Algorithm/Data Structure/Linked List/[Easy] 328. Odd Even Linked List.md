# 328. Odd Even Linked List 

## Problem Description 

**English**:
Given the head of a singly linked list, group all the nodes with odd indices together followed by the nodes with even indices, and return the reordered list.

**中文**:
給定單鏈表的頭結點，將所有奇數索引的節點組合在一起，然後是所有偶數索引的節點，並返回重新排列的鏈表。

## Solution Explanation 

**English**:
To solve this problem, we need to traverse the linked list and separate the nodes into two lists: one for odd indices and one for even indices. We then link the odd list to the even list to form the reordered linked list.

**中文**:
為了解決這個問題，我們需要遍歷鏈表，並將節點分為兩個鏈表：一個是奇數索引，一個是偶數索引。然後我們將奇數鏈表連接到偶數鏈表以形成重新排列的鏈表。

## Detailed Steps 

**English**:
1. **Check if the List is Empty**: If the list is empty, return the head immediately.
2. **Initialize Pointers**: Use pointers `odd` and `even` to traverse the list starting from the head and the second node, respectively. Keep a pointer `even_head` to remember the start of the even list.
3. **The role of `even_head`**: is to store the head of the even-indexed nodes, so that after we rearrange all the odd and even nodes, we can connect the last node of the odd-indexed nodes to the head of the even-indexed nodes, ensuring the entire list is connected correctly.
Without `even_head`, we would not be able to find the starting point of the even-indexed nodes after the rearrangement, and thus would not be able to correctly connect the two groups of nodes.
4. **Traverse the List**:
    - Link the `odd` node to the next odd node.
    - Move the `odd` pointer to the next odd node.
    - Link the `even` node to the next even node.
    - Move the `even` pointer to the next even node.
5. **Link Odd and Even Lists**: After traversal, link the last odd node to the head of the even list.
6. **Return the Head**: Return the head of the modified list.

**中文**:
1. **檢查鏈表是否為空**：如果鏈表為空，立即返回頭結點。
2. **初始化指針**：使用指針 `odd` 和 `even` 分別從頭結點和第二個節點開始遍歷鏈表。使用指針 `even_head` 記住偶數鏈表的開始位置。
3. **`even_head`**: 它的作用是保存偶數節點組的頭節點，這樣當我們重新排列完所有的奇數和偶數節點後，可以將奇數節點組的最後一個節點連接到偶數節點組的頭節點，確保整個鏈表的連接正確。
如果沒有 `even_head`，我們在重新排列後將無法找到偶數節點組的起點，從而無法正確連接兩個節點組。
4. **遍歷鏈表**：
    - 將 `odd` 節點連接到下一個奇數節點。
    - 將 `odd` 指針移動到下一個奇數節點。
    - 將 `even` 節點連接到下一個偶數節點。
    - 將 `even` 指針移動到下一個偶數節點。
5. **連接奇數和偶數鏈表**：遍歷結束後，將最後一個奇數節點連接到偶數鏈表的頭部。
6. **返回頭結點**：返回修改後的鏈表的頭結點。

## Time Complexity / 時間複雜度

**English**:
The time complexity of this solution is **O(n)**, where `n` is the number of nodes in the linked list. This is because each node is visited exactly once.

**中文**:
該解決方案的時間複雜度是 **O(n)**，其中 `n` 是鏈表中的節點數。這是因為每個節點只被訪問一次。

## Visual Explanation / 視覺化解釋

### English:

Suppose we have a linked list:

- `head`: 1 -> 2 -> 3 -> 4 -> 5

#### Initial State
- `odd` points to 1.
- `even` points to 2.
- `even_head` points to 2.

#### First Iteration
- `odd.next = even.next` links 1 to 3. (odd.next=2, even.next=3, odd.next = even.next, so odd.next=3 and so on)
- Move `odd` to 3.
- `even.next = odd.next` links 2 to 4.
- Move `even` to 4.

List after first iteration: 1 -> 3 -> 4 -> 5, `odd` = 3, `even` = 4

#### Second Iteration
- `odd.next = even.next` links 3 to 5.
- Move `odd` to 5.
- `even.next = odd.next` links 4 to None.
- Move `even` to None.

List after second iteration: 1 -> 3 -> 5, `odd` = 5, `even` = None

#### Link Odd and Even Lists
- `odd.next = even_head` links 5 to 2.

Final list: 1 -> 3 -> 5 -> 2 -> 4

### 中文：

假設我們有一個鏈表：

- `head`: 1 -> 2 -> 3 -> 4 -> 5

#### 初始狀態
- `odd` 指向 1。
- `even` 指向 2。
- `even_head` 指向 2。

#### 第一次迴圈
- `odd.next = even.next` 將 1 連接到 3。 (odd.next=2, even.next=3, odd.next = even.next, so odd.next=3 and so on)
- 將 `odd` 移動到 3。
- `even.next = odd.next` 將 2 連接到 4。
- 將 `even` 移動到 4。

第一次迴圈後的鏈表：1 -> 3 -> 4 -> 5，`odd` = 3，`even` = 4

#### 第二次迴圈
- `odd.next = even.next` 將 3 連接到 5。
- 將 `odd` 移動到 5。
- `even.next = odd.next` 將 4 連接到 None。
- 將 `even` 移動到 None。

第二次迴圈後的鏈表：1 -> 3 -> 5，`odd` = 5，`even` = None

#### 連接奇數和偶數鏈表
- `odd.next = even_head` 將 5 連接到 2。

最終鏈表：1 -> 3 -> 5 -> 2 -> 4

## Code Implementation / 代碼實現

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def oddEvenList(self, head):
    if not head:
        return head
    odd = head
    even = head.next
    even_head = even
    while even and even.next:
        odd.next = even.next
        odd = odd.next
        even.next = odd.next
        even = even.next

    odd.next = even_head
    return head

# Example usage:
# List: 1 -> 2 -> 3 -> 4 -> 5
# After reordering: 1 -> 3 -> 5 -> 2 -> 4

def print_list(head):
    current = head
    while current:
        print(current.val, end=" -> " if current.next else "\n")
        current = current.next

# Creating the list 1 -> 2 -> 3 -> 4 -> 5
list_head = ListNode(1, ListNode(2, ListNode(3, ListNode(4, ListNode(5)))))
solution = Solution()
list_head = solution.oddEvenList(list_head)

# Printing the modified list
print_list(list_head)
