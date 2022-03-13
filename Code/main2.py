from API import Requests as Req
from FileHandler import FileHandler as Fh
from Preprocessing import Preprocess as Pre
from FeaturesExtraction import FeatureExtraction as Fe
from Utils import Utils as Utl
from Models import lstm_attention as LSTM_clf


if True:
#if __name__ == '__main__':
    #csv_file_path = "Dataset\dialect_dataset.csv"
    #df = Fh.read_csv(csv_file_path)
    #save_directory_path = "Dataset"
    #Req.get_dataset_df(df, save_directory_path)

    csv_file_path = "Dataset\csv_text_dataset.csv"
    df = Fh.read_arabic_csv(csv_file_path)

    df = Pre.preprocess_df(df, col="Text")

    train, val = Utl.split_dataset(df, y_col="Dialect", test_size=0.06, with_stratify=True, shuffle=True)

    max_statment_len = Fe.get_max_statment_len(train, "Text")
    tokenizer, vocab_size = Fe.get_tokenizer_obj(train["Text"].values)
    directory = "Weights"
    tokenizer_file = "tokenizer.pkl"
    Utl.save_model_pkl(tokenizer, directory, tokenizer_file)
    tokenizer_file = "Weights\\tokenizer.pkl"
    tokenizer = Utl.load_model_pkl(tokenizer_file)

    X_train = Fe.tokenize_texts_to_sequences(tokenizer, train["Text"].values)
    X_train = Fe.padding_sequences(X_train, max_statment_len)

    X_val = Fe.tokenize_texts_to_sequences(tokenizer, val["Text"].values)
    X_val = Fe.padding_sequences(X_val, max_statment_len)

    label_encoder = Utl.get_label_encoder_obj(train["Dialect"])
    directory = "Weights"
    label_encoder_file = "label_encoder.pkl"
    Utl.save_model_pkl(label_encoder, directory, label_encoder_file)
    train["Dialect"] = Utl.get_y_label_encoder(label_encoder, train["Dialect"])
    val["Dialect"] = Utl.get_y_label_encoder(label_encoder, val["Dialect"])

    num_classes = Utl.get_nb_classes(train["Dialect"])

    y_train = Utl.one_hot_encode(train["Dialect"], num_classes)
    y_val = Utl.one_hot_encode(val["Dialect"], num_classes)

    max_text_length = X_train.shape[1]
    directory = "Weights"
    LSTM_clf_params = "lstm_clf_params.json"
    params_dict = {
        "vocab_size": vocab_size,
        "max_text_length": max_text_length,
        "num_classes": num_classes
    }
    Fh.save_json_file(params_dict, directory, LSTM_clf_params)
    LSTM_clf_params = "Weights\lstm_clf_params.json"
    params_dict = Fh.load_json_file(LSTM_clf_params)
    model = LSTM_clf.build_model(params_dict["vocab_size"],
                                 params_dict["max_text_length"],
                                 params_dict["num_classes"],
                                 learning_rate=0.001)
    weights_path = "Weights"
    model, history = LSTM_clf.train_model(model, X_train, y_train,
                                          X_val, y_val,
                                          # class_weights,
                                          weights_path)

    model.load_weights("Weights\lstm_attention_model_weights.h5")
    y_hat = model.predict(X_val)
    Utl.get_prediction_results(y_val, y_hat, label_encoder, num_classes)
