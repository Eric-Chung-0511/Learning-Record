import pandas as pd
from collections import Counter

def find_most_common_words(df, column, num_words=None):
    """
    Find the most common words in a specified column of a DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data.
    column (str): Column name to analyze.
    num_words (int, optional): Number of most common words to return. If None, return all words.

    Returns:
    list: List of tuples containing the most common words and their counts.
    """
    # Split posts into nested lists of words
    nested_words = list(df[column].apply(lambda x: x.split()))
    
    def flatten_list(nested_list):
        """
        Flatten a nested list (a list of lists) into a single list.

        Parameters:
        nested_list (list): Nested list to flatten.

        Returns:
        list: Flattened list.
        """
        flat_list = []
        for post in nested_list:
            for word in post:
                flat_list.append(word)
        return flat_list

    # Flatten the nested list of words
    flat_words = flatten_list(nested_words)

    # Count the occurrences of each word
    word_counts = Counter(flat_words)
    
    # Get the most common words
    if num_words is None:
        most_common_words = word_counts.most_common()
    else:
        most_common_words = word_counts.most_common(num_words)
    
    return most_common_words

# Example usage:
# df = pd.DataFrame({'posts': ["This is a post", "This is another post", "Yet another post"]})
# common_words = find_most_common_words(df, 'posts', 10)
# print(common_words)
