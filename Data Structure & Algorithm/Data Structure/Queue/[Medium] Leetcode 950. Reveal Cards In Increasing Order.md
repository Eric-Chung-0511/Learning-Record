# 950. Reveal Cards In Increasing Order 

## Problem Description 

**English**:
Given an array of integers representing a deck of cards, return an array of integers where the cards are revealed in increasing order. The order of revealing cards is such that you repeatedly reveal the top card, put the next top card at the bottom of the deck, and repeat until all cards are revealed.

**中文**:
給定一個整數數組，表示一副牌的卡片，返回一個整數數組，其中卡片按遞增順序顯示。揭示卡片的順序是反覆揭示頂部的卡片，將下一張頂部的卡片放在牌底，並重複此過程直到所有卡片都被揭示。

## Solution Explanation 

**English**:
To solve this problem, we use a deque to simulate the process of revealing cards in the described order:
1. Sort the deck.
2. Initialize an index deque with indices of the deck.
3. Create a result list to store the revealed cards.
4. For each card in the sorted deck:
   - Place the card in the result list at the position given by the front of the deque.
   - Move the index from the front of the deque to the back to simulate placing the next card at the bottom of the deck.
5. Return the result list.

**中文**:
為了解決這個問題，我們使用雙端隊列來模擬以描述的順序揭示卡片的過程：
1. 將牌排序。
2. 用牌的索引初始化一個索引隊列。
3. 創建一個結果列表來儲存翻開的卡片。
4. 對於排序後的每張牌：
   - 將卡片放置在索引隊列前端指定的位置上。
   - 將索引從隊列前端移到後端，以模擬將下一張卡片放在牌底。
5. 返回結果列表。

## Detailed Steps 

**English**:
1. **Sort the Deck**: Sort the `deck` array in ascending order.
2. **Initialize Index Deque**: Create a deque with indices from `0` to `n-1`.
3. **Create Result List**: Initialize a result list `res` with the same length as the deck, filled with zeros.
4. **Simulate the Process**:
    - For each card in the sorted deck:
        - Place the card at the position specified by the front of the index deque.
        - Move the index from the front to the back of the deque.
5. **Return Result**: Return the `res` list.

**中文**:
1. **排序牌**：按升序排序 `deck` 數組。
2. **初始化索引隊列**：創建一個包含從 `0` 到 `n-1` 的索引的雙端隊列。
3. **創建結果列表**：初始化與牌長度相同的結果列表 `res`，填充為零。
4. **模擬過程**：
    - 對於排序後的每張卡片：
        - 將卡片放置在索引隊列前端指定的位置上。
        - 將索引從隊列前端移到後端。
5. **返回結果**：返回 `res` 列表。

## Time Complexity 

**English**:
The time complexity of this solution is **O(n log n)** due to the sorting step. The subsequent operations of deque manipulation are **O(n)**.

**中文**:
該解決方案的時間複雜度是 **O(n log n)**，因為排序步驟的時間複雜度是 **O(n log n)**。後續操作的時間複雜度是 **O(n)**。

## Visual Explanation 

### English:

Suppose we have a deck: `[17, 13, 11, 2, 3, 5, 7]`

1. **Sort the Deck**: `[2, 3, 5, 7, 11, 13, 17]`
2. **Initialize Index Deque**: `deque([0, 1, 2, 3, 4, 5, 6])`
3. **Create Result List**: `[0, 0, 0, 0, 0, 0, 0]`

#### First Iteration (card = 2)
- Place 2 at `res[0]`.
- `res`: `[2, 0, 0, 0, 0, 0, 0]`
- Move index 0 to the back.
- `deque([1, 2, 3, 4, 5, 6, 0])`

#### Second Iteration (card = 3)
- Place 3 at `res[1]`.
- `res`: `[2, 3, 0, 0, 0, 0, 0]`
- Move index 1 to the back.
- `deque([2, 3, 4, 5, 6, 0, 1])`

Continue this process for the remaining cards.

Final result: `[2, 13, 3, 11, 5, 17, 7]`

### 中文：

假設我們有一副牌：`[17, 13, 11, 2, 3, 5, 7]`

1. **排序牌**：`[2, 3, 5, 7, 11, 13, 17]`
2. **初始化索引隊列**：`deque([0, 1, 2, 3, 4, 5, 6])`
3. **創建結果列表**：`[0, 0, 0, 0, 0, 0, 0]`

#### 第一次迭代（卡片 = 2）
- 將 2 放在 `res[0]`。
- `res`：`[2, 0, 0, 0, 0, 0, 0]`
- 將索引 1 移到後面。
- `deque([2, 3, 4, 5, 6, 1])`

#### 第二次迭代（卡片 = 3）
- 將 3 放在 `res[2]`。
- `res`：`[2, 0, 3, 0, 0, 0, 0]`
- 將索引 3 移到後面。
- `deque([4, 5, 6, 1, 3])`

#### 第三次迭代（卡片 = 5）
- 將 5 放在 `res[4]`。
- `res`：`[2, 0, 3, 0, 5, 0, 0]`
- 將索引 5 移到後面。
- `deque([6, 1, 3, 5])`

#### 第四次迭代（卡片 = 7）
- 將 7 放在 `res[6]`。
- `res`：`[2, 0, 3, 0, 5, 0, 7]`
- 將索引 1 移到後面。
- `deque([3, 5, 1])`

繼續對剩餘卡片進行此過程。

最終結果：`[2, 13, 3, 11, 5, 17, 7]`

## Code Implementation 

```python
class Solution:
    def deckRevealedIncreasing(self, deck):
        n = len(deck)
        index = deque(range(n))
        res = [0] * n

        sorted_card = sorted(deck)
        for card in sorted_card:
            res[index.popleft()] = card
            if index:
                index.append(index.popleft())
        return res

# Example usage:
# deck = [17, 13, 11, 2, 3, 5, 7]
# The order of cards revealed in increasing order: [2, 13, 3, 11, 5, 17, 7]

solution = Solution()
result = solution.deckRevealedIncreasing([17, 13, 11, 2, 3, 5, 7])
print(result)  # Output: [2, 13, 3, 11, 5, 17, 7]
