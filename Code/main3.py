from API import Requests as Req
from FileHandler import FileHandler as Fh
from Preprocessing import Preprocess as Pre
from FeaturesExtraction import FeatureExtraction as Fe
from Utils import Utils as Utl
from Models import basic_bert_model as BERT_clf

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

    label_encoder = Utl.get_label_encoder_obj(train["Dialect"])
    train["Dialect"] = Utl.get_y_label_encoder(label_encoder, train["Dialect"])
    val["Dialect"] = Utl.get_y_label_encoder(label_encoder, val["Dialect"])

    max_statment_len = Fe.get_max_statment_len(train, "Text")

    train_inp, train_mask, train_label = BERT_clf.tokenizer_encode(train["Text"], train["Dialect"], max_statment_len)
    val_inp, val_mask, val_label = BERT_clf.tokenizer_encode(val["Text"], val["Dialect"], max_statment_len)

    num_classes = Utl.get_nb_classes(train["Dialect"])
    bert_model = BERT_clf.build_bert_model(num_classes)
    weights_dir = "Weights"
    bert_model, history = BERT_clf.fit_bert_model(bert_model, train_inp, train_mask, train_label,
                                                  val_inp, val_mask, val_label, weights_dir)