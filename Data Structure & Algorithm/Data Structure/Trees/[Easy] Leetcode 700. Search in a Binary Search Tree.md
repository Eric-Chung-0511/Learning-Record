# 700. Search in a Binary Search Tree 

## Problem Description 
**English**:
You are given the root of a binary search tree (BST) and an integer `val`.

Find the node in the BST where the node's value equals `val` and return the subtree rooted with that node. If such a node does not exist, return `null`.

**中文**:
給定二叉搜索樹（BST）的根節點和一個整數 `val`。

找到 BST 中值等於 `val` 的節點，並返回以該節點為根的子樹。如果不存在這樣的節點，則返回 `null`。

## Solution Explanation 

**English**:
To solve this problem, we can utilize the properties of a BST where the left subtree contains values less than the root and the right subtree contains values greater than the root. We can perform a binary search to find the node with the given value.

**中文**:
為了解決這個問題，我們可以利用 BST 的特性，其中左子樹包含小於根的值，右子樹包含大於根的值。我們可以執行二分搜索來找到具有給定值的節點。

## Detailed Steps 

**English**:
1. **Check the Root**: If the root is `null`, return `null`.
2. **Compare with Root**:
    - If the value of the root node is equal to `val`, return the root.
    - If the value of `val` is less than the root node's value, search in the left subtree.
    - If the value of `val` is greater than the root node's value, search in the right subtree.
3. **Return the Result**: Continue the search until the node is found or return `null` if the node does not exist.

**中文**:
1. **檢查根節點**：如果根節點為 `null`，返回 `null`。
2. **與根節點比較**：
    - 如果根節點的值等於 `val`，返回根節點。
    - 如果 `val` 小於根節點的值，在左子樹中搜索。
    - 如果 `val` 大於根節點的值，在右子樹中搜索。
3. **返回結果**：繼續搜索直到找到節點或如果節點不存在則返回 `null`。

## Time Complexity 
**English**:
The time complexity of this solution is **O(h)**, where `h` is the height of the tree. In the worst case, this could be **O(n)** for a skewed tree.

**中文**:
該解決方案的時間複雜度是 **O(h)**，其中 `h` 是樹的高度。在最壞的情況下，對於一棵偏斜樹，這可能是 **O(n)**。

## Code Implementation

```python
class Solution:
    def searchBST(self, root, val):
        if not root:
            return None
        
        if root.val == val:
            return root
        elif val < root.val:
            return self.searchBST(root.left, val)
        else:
            return self.searchBST(root.right, val)

