import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, \
    classification_report, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score as f1_score_rep
import keras.backend as K
from sklearn.utils import class_weight
from keras.utils.np_utils import to_categorical
from numpy import unique, newaxis
from Visualization.Visualization import ROC_plot
from keras.models import load_model
import joblib
import os


from sklearn.model_selection import train_test_split


def split_dataset(df, y_col="", test_size=0.20, with_stratify=True, shuffle=True):
    if with_stratify:
        train, val = train_test_split(df,
                                      test_size=test_size,
                                      random_state=1,
                                      stratify=df[y_col],
                                      shuffle=shuffle)
    else:
        train, val = train_test_split(df,
                                      test_size=test_size,
                                      random_state=1,
                                      stratify=df[y_col],
                                      shuffle=shuffle)
    return train, val


def get_label_encoder_obj(y):
    label_encoder = LabelEncoder()
    return label_encoder.fit(y)


def get_y_label_encoder(label_encoder, y):
    return label_encoder.transform(y)


def get_nb_classes(y):
    return len(unique(y))


def one_hot_encode(y, num_classes):
    return to_categorical(y, num_classes=num_classes)


def f1_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val


def get_class_weights(y):
    class_weights = class_weight.compute_class_weight('balanced',
                                                      classes=unique(y),
                                                      y=y)
    return {k: v for k, v in enumerate(class_weights)}


def print_score(y_pred, y_real, label_encoder):
    print("Accuracy: ", accuracy_score(y_real, y_pred))
    print("Precision:: ", precision_score(y_real, y_pred, average="micro"))
    print("Recall:: ", recall_score(y_real, y_pred, average="micro"))
    print("F1_Score:: ", f1_score_rep(y_real, y_pred, average="micro"))

    print()
    print("Macro precision_recall_fscore_support (macro) average")
    print(precision_recall_fscore_support(y_real, y_pred, average="macro"))

    print()
    print("Macro precision_recall_fscore_support (micro) average")
    print(precision_recall_fscore_support(y_real, y_pred, average="micro"))

    print()
    print("Macro precision_recall_fscore_support (weighted) average")
    print(precision_recall_fscore_support(y_real, y_pred, average="weighted"))

    print()
    print("Confusion Matrix")
    cm = confusion_matrix(y_real, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, newaxis]
    df_cm = pd.DataFrame(cm, index=[i for i in label_encoder.classes_],
                         columns=[i for i in label_encoder.classes_])
    plt.figure(figsize=(20, 20))
    sns.heatmap(df_cm, annot=True)

    print()
    print("Classification Report")
    print(classification_report(y_real, y_pred, target_names=label_encoder.classes_))


def get_prediction_results(y_true, y_hat, label_encoder, num_classes):
    if len(y_true.shape) == 1:
        y_train_ohe = one_hot_encode(y_true, num_classes)
        y_hat_ohe = one_hot_encode(y_hat, num_classes)
    else:
        y_train_ohe = y_true.copy()
        y_hat_ohe = y_hat.copy()
    print(y_hat[:5])
    ROC_plot(y_train_ohe, y_hat_ohe, label_encoder, num_classes)
    print_score(y_hat, y_true, label_encoder)


def predict(model, X_val):
    return model.predict(X_val)


def save_model_pkl(model, path_directory, file_name):
    joblib.dump(model, os.path.join(path_directory, file_name))


def load_model_pkl(file_directory):
    return joblib.load(file_directory)


def load_model_weights(model, file_directory):
    model.load_weights(file_directory)
    return model


def check_file_exists(file_path):
    if os.path.exists(file_path):
        return file_path
    raise Exception("Sorry, No file exists with this path")
