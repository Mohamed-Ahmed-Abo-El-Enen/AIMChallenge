from FileHandler import FileHandler as Fh
from API import Requests as Req
from Preprocessing import Preprocess as Pre
from Visualization import Visualization as Vs
from FeaturesExtraction import FeatureExtraction as Fe
from Utils import Utils as Utl
from Models import SGDClassifier_Model as SGD_clf
from Models import SVMClassifier_Model as SVM_clf


if True:
#if __name__ == '__main__':
    #csv_file_path = "Dataset\dialect_dataset.csv"
    #df = Fh.read_csv(csv_file_path)
    #save_directory_path = "Dataset"
    #Req.get_dataset_df(df, save_directory_path)

    csv_file_path = "Dataset\csv_text_dataset.csv"
    df = Fh.read_arabic_csv(csv_file_path)

    df = Pre.preprocess_df(df, col="Text")

    Vs.generate_word_cloud(df, "Text")
    Vs.most_freq_word(df, "Text")

    Vs.get_classes_freq(df, "Dialect")
    Vs.get_classes_percentage(df, "Dialect")

    train, val = Utl.split_dataset(df, y_col="Dialect", test_size=0.06, with_stratify=True, shuffle=True)

    X_train = train["Text"].values
    y_train = train["Dialect"].values
    X_val = val["Text"].values
    y_val = val["Dialect"].values

    label_encoder = Utl.get_label_encoder_obj(y_train)
    y_train = Utl.get_y_label_encoder(label_encoder, y_train)
    y_val = Utl.get_y_label_encoder(label_encoder, y_val)

    num_classes = Utl.get_nb_classes(y_train)

    count_vect = Fe.CountVectorizer_fit(X_train, ngram_range=(1, 2))
    X_train_counts = Fe.CountVectorizer_transform(count_vect, X_train)
    X_val_counts = Fe.CountVectorizer_transform(count_vect, X_val)

    tf_transformer = Fe.TfidfTransformer_fit(X_train_counts)
    X_train_tfidf = Fe.TfidfTransformer_transform(tf_transformer, X_train_counts)
    X_val_tfidf = Fe.TfidfTransformer_transform(tf_transformer, X_val_counts)

    preprocessing_pipeline = Fe.fit_preprocessing_pipeline(X_train, ngram_range=(1, 2), use_idf=True)
    directory = "Weights"
    preprocessing_pipeline_file = "preprocessing_pipeline.pkl"
    Utl.save_model_pkl(preprocessing_pipeline, directory, preprocessing_pipeline_file)
    preprocessing_pipeline_file = "Weights\preprocessing_pipeline.pkl"
    preprocessing_pipeline = Utl.load_model_pkl(preprocessing_pipeline_file)

    X_train_tfidf = Fe.transform_preprocessing_pipeline(preprocessing_pipeline, X_train)
    X_val_tfidf = Fe.transform_preprocessing_pipeline(preprocessing_pipeline, X_val)

    model, _ = SGD_clf.fit_SGDClassifier(X_train_tfidf, y_train)

    directory = "Weights"
    SGD_clf_file = "SGD_clf.pkl"
    Utl.save_model_pkl(model, directory, SGD_clf_file)
    SGD_clf_file = "Weights\SGD_clf.pkl"
    model = Utl.load_model_pkl(SGD_clf_file)

    #y_hat = model.predict(X_val_tfidf)
    #Utl.get_prediction_results(y_val, y_hat, label_encoder, num_classes)
    #y_hat = Utl.predict(model, X_val_tfidf)

    model, _ = SVM_clf.fit_SVMClassifier(X_train_tfidf, y_train)
    directory = "Weights"
    SVM_clf_file = "SVM_clf.pkl"
    Utl.save_model_pkl(model, directory, SVM_clf_file)
    SVM_clf_file = "Weights\SVM_clf.pkl"
    model = Utl.load_model_pkl(SVM_clf_file)

    y_hat = model.predict(X_val_tfidf)
    Utl.get_prediction_results(y_val, y_hat, label_encoder, num_classes)
    y_hat = Utl.predict(model, X_val_tfidf)