from tensorflow.keras.optimizers import Adam
from keras.layers import Layer
from keras.layers import LSTM, Embedding, Dense, Input, Dropout, Bidirectional, Reshape, Permute, Lambda
from keras.layers import Flatten
from keras.models import Model
from keras.metrics import Recall, Precision
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from livelossplot import PlotLossesKeras
from os.path import join
import keras.backend as K
import time


def f1_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val


class attention(Layer):
    def __init__(self, return_sequences=True, layer_name=""):
        self.layer_name = layer_name
        self.return_sequences = return_sequences
        super(attention, self).__init__()

    def build(self, input_shape):
        self.W=self.add_weight(name=self.layer_name+"_att_weight", 
                               shape=(input_shape[-1],1),
                               initializer="normal")
        self.b=self.add_weight(name=self.layer_name+"_att_bias", 
                               shape=(input_shape[1],1),
                               initializer="zeros")
        super(attention, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        if self.return_sequences:
            return output
        return K.sum(output, axis=1)


def build_model(MAX_NB_WORDS,
                MAX_TEXT_LEN,
                nb_classes,
                learning_rate=2e-5,
                epsilon=1e-08):
    EMBEDDING_DIM = 100
    inputs = Input(name='inputs', shape=[MAX_TEXT_LEN])
    layer = Embedding(MAX_NB_WORDS+1, EMBEDDING_DIM, input_length=MAX_TEXT_LEN)(inputs)

    #lstm_units = [64, 32]
    #for units in lstm_units:
    #    layer = Bidirectional(LSTM(units, return_sequences=True))(layer)
    #    layer = attention(layer_name="attention_"+str(units),return_sequences=True)(layer)
        #layer = Dropout(0.3)(layer)

    units = 64
    layer = Bidirectional(LSTM(units, return_sequences=True))(layer)
    layer = attention(layer_name="attention_"+str(units), 
                      return_sequences=False)(layer)
    #layer = Dropout(0.4)(layer)

    layer = Flatten()(layer)
    output = Dense(nb_classes, activation='softmax')(layer)#, 
                  #kernel_regularizer='l1')(layer)
    model = Model(inputs=[inputs], outputs=output)
    model.compile(optimizer=Adam(learning_rate=learning_rate, epsilon=epsilon),
                  loss='categorical_crossentropy', metrics=['accuracy', Precision(), Recall(), f1_score])
    print(model.summary())
    return model


def train_model(model, X_train, y_train, X_val, y_val,
                #class_weights,
                weights_dir,
                epochs=20,
                mini_batch_size=32,
                ):
    model_weights_file_path = join(weights_dir, "lstm_attention_model_weights.h5")
    checkpoint = ModelCheckpoint(filepath=model_weights_file_path, monitor="val_accuracy", verbose=1, save_best_only=True, mode="max", save_weights_only=True)
    early_stopping = EarlyStopping(monitor="val_accuracy", mode="max", verbose=1, patience=5)
    lr_reduce = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, verbose=0, mode='max', min_delta=0.0001, cooldown=0, min_lr=0)
    plotlosses = PlotLossesKeras()
    call_backs = [checkpoint, early_stopping, lr_reduce, plotlosses]
    start_time = time.time()

    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=epochs,
                        batch_size=mini_batch_size,
                        callbacks=call_backs,
                        #class_weight=class_weights,
                        verbose=1)

    duration = time.time() - start_time
    print("Model take {} S to train ".format(duration))
    return model, history
