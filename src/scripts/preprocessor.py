import re

def normalize_amharic(text):
    """
    Normalize Amharic text by removing redundant characters.

    """
    subs = {
        '[ሀሐኀሃኃ]': 'ሀ', '[ሁሑኁ]': 'ሁ', '[ሂሑኂ]': 'ሂ', '[ሄሔኄ]': 'ሄ', '[ህሕኅ]': 'ህ', '[ሆሖኆ]': 'ሆ',
        '[ሠሰ]': 'ሰ', '[ሡሱ]': 'ሱ', '[ሢሲ]': 'ሲ', '[ሣሳ]': 'ሳ', '[ሤሴ]': 'ሴ', '[ሥስ]': 'ስ', '[ሦሶ]': 'ሶ',
        '[ዐአዓኣ]': 'አ', '[ዑኡ]': 'ኡ', '[ዒኢ]': 'ኢ', '[ዔኤ]': 'ኤ', '[ዕእ]': 'እ', '[ዖኦ]': 'ኦ',
        '[ጸፀ]': 'ጸ', '[ፁጹ]': 'ጹ', '[ፂጺ]': 'ጺ', '[ፃጻ]': 'ጻ', '[ፄጼ]': 'ጼ', '[ፅጽ]': 'ጽ', '[ጾፆ]': 'ጾ',
        '[ሗኈ]': 'ኈ', '[ሧሷ]': 'ሷ'
    }
    for pattern, replacement in subs.items():
        text = re.sub(pattern, replacement, text)
    return text.strip()

def remove_emojies(text):
        
    """
    Removes emoji characters from the input text string.
    This function uses a regular expression to identify and remove a wide range of emoji characters,
    including emoticons, symbols, pictographs, transport and map symbols, flags, dingbats, enclosed characters,
    and supplemental symbols and pictographs.
    Args:
        text (str): The input string from which emojis should be removed.
    Returns:
        str: The input string with all emoji characters removed and leading/trailing whitespace stripped.
    """

    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # Emoticons
        "\U0001F300-\U0001F5FF"  # Symbols & pictographs
        "\U0001F680-\U0001F6FF"  # Transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # Flags
        "\U00002700-\U000027BF"  # Dingbats
        "\U000024C2-\U0001F251"  # Enclosed characters
        "\U0001F900-\U0001F9FF"  # Supplemental symbols and pictographs
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text).strip()

def tokenize(text):
    # Remove extra punctuation
    text = re.sub(r'[\"“”።፣.,!?…()]', ' ', text)
    # Split on whitespace
    tokens = text.strip().split()
    return tokens

if __name__ == "__main__":
    sample_text =  "ዋጋ 1000 ብር 😊📦 በአሠዲስ አበባ 🚚🥮"
    print(remove_emojies(sample_text))