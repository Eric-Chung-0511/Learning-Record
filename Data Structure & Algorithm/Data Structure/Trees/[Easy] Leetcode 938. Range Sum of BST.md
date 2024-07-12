# 938. Range Sum of BST 

## Problem Description 

**English**:
Given the root of a binary search tree (BST) and two integers `low` and `high`, return the sum of values of all nodes with a value in the inclusive range `[low, high]`.

**中文**:
給定二叉搜索樹（BST）的根節點和兩個整數 `low` 和 `high`，返回值在 `[low, high]` 包含範圍內的所有節點值的總和。

## Solution Explanation 

**English**:
To solve this problem, we perform a depth-first search (DFS) traversal of the BST and sum the values of nodes that fall within the range `[low, high]`. We can take advantage of the BST properties to optimize the search:
- If the current node's value is greater than `low`, continue to search the left subtree.
- If the current node's value is less than `high`, continue to search the right subtree.
- If the current node's value is within the range `[low, high]`, add its value to the sum.

**中文**:
為了解決這個問題，我們執行深度優先搜索（DFS）遍歷 BST，並將值在 `[low, high]` 範圍內的節點值相加。我們可以利用 BST 的特性來優化搜索：
- 如果當前節點的值大於 `low`，繼續搜索左子樹。
- 如果當前節點的值小於 `high`，繼續搜索右子樹。
- 如果當前節點的值在 `[low, high]` 範圍內，將其值加到總和中。

## Detailed Steps 

**English**:
1. **Check the Root**: If the root is `null`, return 0.
2. **Initialize Sum**: Initialize a variable `sum` to store the sum of node values within the range.
3. **Traverse the Tree**:
    - If the current node's value is within the range `[low, high]`, add its value to `sum`.
    - If the current node's value is greater than `low`, traverse the left subtree.
    - If the current node's value is less than `high`, traverse the right subtree.
4. **Return the Sum**: Return the value of `sum`.

**中文**:
1. **檢查根節點**：如果根節點為 `null`，返回 0。
2. **初始化總和**：初始化變量 `sum` 來存儲範圍內節點值的總和。
3. **遍歷樹**：
    - 如果當前節點的值在 `[low, high]` 範圍內，將其值加到 `sum` 中。
    - 如果當前節點的值大於 `low`，遍歷左子樹。
    - 如果當前節點的值小於 `high`，遍歷右子樹。
4. **返回總和**：返回 `sum` 的值。

## Time Complexity 
**English**:
The time complexity of this solution is **O(n)**, where `n` is the number of nodes in the tree. This is because, in the worst case, we may need to visit all the nodes in the tree.

**中文**:
該解決方案的時間複雜度是 **O(n)**，其中 `n` 是樹中的節點數量。這是因為在最壞的情況下，我們可能需要訪問樹中的所有節點。

## Code Implementation

```python
class Solution:
    def rangeSumBST(self, root, low, high):
        if not root:
            return 0

        sum = 0

        if low <= root.val <= high:
            sum += root.val

        if root.val > low:
            sum += self.rangeSumBST(root.left, low, high)

        if root.val < high:
            sum += self.rangeSumBST(root.right, low, high)
