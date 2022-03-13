import numpy as np
from keras.preprocessing.sequence import pad_sequences


def padding_sequences(x_arr, max_len):
    x_arr = pad_sequences(x_arr, maxlen=max_len, value=0, padding='post')
    return x_arr


def predict_sgd_svm_class(X, label_encoder, preprocess, model):
    tmp_x = preprocess.transform([X])
    y_hat = model.predict(tmp_x)
    y_hat = label_encoder.inverse_transform(y_hat)[0]
    return str(y_hat)


def predict_lstm_attention_class(X, label_encoder, preprocess, LSTM_attention_model, model_params=None):
    tmp_x = preprocess.texts_to_sequences([X])
    tmp_x = padding_sequences(tmp_x, model_params["max_text_length"])
    y_hat = np.argmax(LSTM_attention_model.predict(tmp_x), axis=1)
    y_hat = label_encoder.inverse_transform(y_hat)[0]
    return str(y_hat)