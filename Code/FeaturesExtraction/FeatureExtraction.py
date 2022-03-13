from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
import nltk

nltk.download('punkt')
nltk.download('wordnet')

oov_tok = "<oov_tok>"


def CountVectorizer_fit(X_train, ngram_range=(1,1)):
    count_vect = CountVectorizer(ngram_range=ngram_range)
    return count_vect.fit(X_train)


def CountVectorizer_transform(count_vect, X):
    return count_vect.transform(X)


def TfidfTransformer_fit(X_train_counts, use_idf=True):
    tf_transformer = TfidfTransformer(use_idf=use_idf)
    return tf_transformer.fit(X_train_counts)


def TfidfTransformer_transform(tf_transformer, X_counts):
    return tf_transformer.transform(X_counts)


def fit_preprocessing_pipeline(X_train, ngram_range=(1,1), use_idf=True):
    pipeline = Pipeline([
        ("vect", CountVectorizer(ngram_range=ngram_range)),
        ("tfidf", TfidfTransformer(use_idf=use_idf))])
    pipeline.fit(X_train)
    return pipeline


def transform_preprocessing_pipeline(pipeline, X):
    return pipeline.transform(X)


def get_max_sequences_len(df, col):
    return max([len(x.split()) for x in df[col].values])


def get_tokenizer_obj(text_list):
    tokenizer = Tokenizer(lower=True, split=" ", oov_token=oov_tok)
    tokenizer.fit_on_texts(text_list)
    return tokenizer, len(tokenizer.word_index)


def tokenize_texts_to_sequences(tokenizer, text_list):
    return tokenizer.texts_to_sequences(text_list)


def padding_sequences(x_arr, max_len):
    x_arr = pad_sequences(x_arr, maxlen=max_len, value=0, padding='post')
    return x_arr


def get_max_statment_len(df, col):
    return max([len(text.split()) for text in df[col]])