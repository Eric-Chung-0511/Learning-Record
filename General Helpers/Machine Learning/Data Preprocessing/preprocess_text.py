import pandas as pd
import re

def preprocess_text(df, text_column='posts', remove_special=True, special_words=None):
    """
    Preprocess text data in a DataFrame column.

    Parameters:
    df (pd.DataFrame): DataFrame containing the text data.
    text_column (str): Name of the column containing the text data. Default is 'posts'.
    remove_special (bool): Whether to remove special words. Default is True.
    special_words (list): List of special words to remove. Default is None.

    Returns:
    pd.DataFrame: DataFrame with the preprocessed text data.
    """
    # Copy the text data
    texts = df[text_column].copy()

    # Remove links
    df[text_column] = df[text_column].apply(lambda x: re.sub(r'https?:\/\/.*?[\s+]', '', x.replace("|"," ") + " "))
    
    # Keep the End Of Sentence characters
    df[text_column] = df[text_column].apply(lambda x: re.sub(r'\.', ' EOSTokenDot ', x + " "))
    df[text_column] = df[text_column].apply(lambda x: re.sub(r'\?', ' EOSTokenQuest ', x + " "))
    df[text_column] = df[text_column].apply(lambda x: re.sub(r'!', ' EOSTokenExs ', x + " "))
    
    # Strip punctuation
    df[text_column] = df[text_column].apply(lambda x: re.sub(r'[\.+]', ".", x))

    # Remove multiple full stops
    df[text_column] = df[text_column].apply(lambda x: re.sub(r'[^\w\s]','', x))

    # Remove non-words
    df[text_column] = df[text_column].apply(lambda x: re.sub(r'[^a-zA-Z\s]','', x))

    # Convert posts to lowercase
    df[text_column] = df[text_column].apply(lambda x: x.lower())

    # Remove multiple letter repeating words
    df[text_column] = df[text_column].apply(lambda x: re.sub(r'([a-z])\1{2,}[\s|\w]*','', x)) 

    # Remove very short or long words
    df[text_column] = df[text_column].apply(lambda x: re.sub(r'\b\w{1,3}\b','', x)) 
    df[text_column] = df[text_column].apply(lambda x: re.sub(r'\b\w{30,1000}\b','', x))

    # Remove special words if specified
    if remove_special and special_words:
        special_words = [word.lower() for word in special_words]
        p = re.compile("(" + "|".join(special_words) + ")")
        df[text_column] = df[text_column].apply(lambda x: p.sub('', x))
    
    return df

# Example usage:
# special_words_list = ['example', 'special', 'words']
# new_df = preprocess_text(data, text_column='posts', special_words=special_words_list)
# print(new_df.head())
