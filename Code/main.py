from app.app import *
from Utils.Utils import *
from Models import lstm_attention as LSTM_clf
from FileHandler.FileHandler import load_json_file

if __name__ == "__main__":
    app.config["encoded_class"] = load_model_pkl(check_file_exists("Weights\label_encoder.pkl"))

    app.config["preprocessing_pipeline"] = load_model_pkl(check_file_exists("Weights\preprocessing_pipeline.pkl"))
    app.config["SGD_clf"] = load_model_pkl(check_file_exists("Weights\SGD_clf.pkl"))
    #app.config["SVM_clf"] = load_model_pkl(check_file_exists("Weights\SVM_clf.pkl"))

    app.config["tokenizer"] = load_model_pkl(check_file_exists("Weights\\tokenizer.pkl"))
    LSTM_clf_params = load_json_file(check_file_exists("Weights\lstm_clf_params.json"))
    app.config["LSTM_clf_params"] = LSTM_clf_params
    lstm_model = LSTM_clf.build_model(LSTM_clf_params["vocab_size"],
                                      LSTM_clf_params["max_text_length"],
                                      LSTM_clf_params["num_classes"])
    lstm_model = load_model_weights(lstm_model, check_file_exists("Weights\lstm_attention_model_weights.h5"))
    app.config["lstm_model"] = lstm_model

    app.run()