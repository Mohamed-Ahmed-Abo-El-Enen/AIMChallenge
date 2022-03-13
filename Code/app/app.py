from flask import Flask, render_template, request
from .Utils.Utils import *
from .Utils.Preprocess import *


def get_model_transformers(classifier_type):
    if classifier_type == "SGD" or classifier_type == "SVM":
        label_encoder = app.config["encoded_class"]
        preprocess = app.config["preprocessing_pipeline"]
        model = app.config["SGD_clf"]
    else:
        label_encoder = app.config["encoded_class"]
        preprocess = app.config["tokenizer"]
        model = app.config["lstm_model"]

    return label_encoder, preprocess, model


def predict_sentance_dialect(X, label_encoder, preprocess, model, classifier_type):
    X = preprocess_text_sample(X)
    if classifier_type == "SGD" or classifier_type == "SVM":
        return predict_sgd_svm_class(X, label_encoder, preprocess, model)
    else:
        return predict_lstm_attention_class(X, label_encoder, preprocess, model, app.config["LSTM_clf_params"])


app = Flask(__name__)

var_dict = {"res":"",
            "text_val":"",
            "classifier_type":"SGD"}


@app.route('/', methods=["GET", "POST"])
def run():
    request_type_str = request.method
    if request_type_str == "GET":
        return render_template("index.html",
                               res=var_dict["res"],
                               text_val=var_dict["text_val"],
                               classifier_type=var_dict["classifier_type"])
    else:
        var_dict["text_val"] = request.form['text']
        var_dict["classifier_type"] = request.form['classifier_type']
        label_encoder, pipeline, model = get_model_transformers(var_dict["classifier_type"])
        var_dict["res"] = predict_sentance_dialect(var_dict["text_val"], label_encoder, pipeline, model,
                                                   var_dict["classifier_type"])
        return render_template("index.html",
                               res=var_dict["res"],
                               text_val=var_dict["text_val"],
                               classifier_type=var_dict["classifier_type"])

