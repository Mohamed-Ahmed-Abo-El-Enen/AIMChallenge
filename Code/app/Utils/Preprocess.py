import re
from string import punctuation
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")
import arabicstopwords.arabicstopwords as stp
punctuation += '،؛؟”“'
stop_words = set(stopwords.words('english'))


def remove_emoji(text):
    regex_pattern = re.compile("["
                                u"\U0001F600-\U0001F64F"  
                                u"\U0001F300-\U0001F5FF"  
                                u"\U0001F680-\U0001F6FF"  
                                u"\U0001F1E0-\U0001F1FF"  
                                u"\U00002500-\U00002BEF"  
                                u"\U00002702-\U000027B0"
                                u"\U00002702-\U000027B0"
                                u"\U000024C2-\U0001F251"
                                u"\U0001f926-\U0001f937"
                                u"\U00010000-\U0010ffff"
                                u"\u2640-\u2642" 
                                u"\u2600-\u2B55"
                                u"\u200d"
                                u"\u23cf"
                                u"\u23e9"
                                u"\u231a"
                                u"\ufe0f"  
                                u"\u3030"
                               "]+", flags=re.UNICODE)

    return regex_pattern.sub(r'', text)


def remove_email(text):
    return re.sub('([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})', '', text)


def remove_repeated_char(text):
    return re.sub(r'(.)\1\1{1,}', r'\1\1', text)


def remove_account_tag(text):
    return re.sub(r'@[\w]+', '', text)


def remove_hashtag(text):
    return re.sub(r'#[\w]+', '', text)


def remove_links(text):
    return re.sub(r'http[^\s]+', '', text)


def remove_spaces(text):
    text = re.sub(r"\n+", ' ', text)
    text = re.sub(r"\t+", ' ', text)
    text = re.sub(r"\r+", ' ', text)
    text = re.sub(r"\s+", ' ', text)
    return text


def remove_tashkeel(text):
    regx_pattern = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
    text = re.sub(regx_pattern, "", text)

    regx_pattern = re.compile(r'(.)\1+')
    subst = r"\1\1"
    text = re.sub(regx_pattern, subst, text)
    return re.sub(r"[^\w\s]", '', text)


def remove_punctuation(text):
    return ''.join(c for c in text if c not in punctuation)


def remove_stop_words(text):
    text_list = []
    for w in text.split():
        if (not stp.is_stop(w)) and (w not in stop_words):
            text_list.append(w)
    return " ".join(text_list)


def remove_less_2_characters(text):
    return re.sub(r"\W*\b\w{1,2}\b", '', text)


def preprocess_text_sample(text):
    text = text.lower()
    text = remove_emoji(text)
    text = remove_email(text)
    text = remove_account_tag(text)
    text = remove_hashtag(text)
    text = remove_links(text)
    text = remove_less_2_characters(text)
    text = remove_repeated_char(text)
    text = remove_punctuation(text)
    text = remove_tashkeel(text)
    text = remove_stop_words(text)
    text = remove_spaces(text)
    text = text.strip()
    return text


def preprocess_text_cols(df, col):
    df[col] = df[col].apply(lambda x: preprocess_text_sample(x))
    return df


def preprocess_df(df, col="Text"):
    df = preprocess_text_cols(df, col)
    df = df[df[col] != ""]
    df.dropna(inplace=True)
    return df