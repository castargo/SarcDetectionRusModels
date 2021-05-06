import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical


class BiLSTMClassifier():
    def __init__(self, input_shape):

        inputs = layers.Input(shape=input_shape)
        
        x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(inputs)
        x = layers.Dropout(0.75)(x)
        x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)
        x = layers.Dropout(0.75)(x)
        x = layers.Bidirectional(layers.LSTM(256))(x)
        x = layers.Dropout(0.75)(x)
        x = layers.Dense(128, activation="relu")(x)
        outputs = layers.Dense(1, activation="sigmoid", name="predictions")(x)
        
        self.model = keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(
            optimizer='RMSprop',
            loss='binary_crossentropy',
            metrics=[keras.metrics.Precision(), keras.metrics.Recall()]
            )
    
    def fit(self, X, y, X_test=None, y_test=None, epochs=10, verbose=0):
        if not X_test is None and not y_test is None:
            self.model.fit(X, y, epochs=epochs, validation_data=(X_test, y_test), verbose=verbose)
        else:
            self.model.fit(X, y, epochs=epochs, verbose=verbose)
        return self
    
    def partial_fit(self, X, y, sample_weight=None, class_weight=None):
        self.model.train_on_batch(X, y, sample_weight=sample_weight, class_weight=class_weight)
        return self
    
    def predict_proba(self, X_test):
        return self.model.predict(X_test)
    
    def predict(self, X_test):
        return self.model.predict(X_test)

    def save(self, path):
        self.model.save(path)

