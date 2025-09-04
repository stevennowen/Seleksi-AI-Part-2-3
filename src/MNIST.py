import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from supervised_learning.ANN import CrossEntropyLoss, Adam

def load_mnist_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train[:1000]  
    y_train = y_train[:1000]
    X_test = X_test[:200]     
    y_test = y_test[:200]
    
    # Normalisasi dan reshape untuk CNN (28x28 -> 32x32 untuk LeNet compatibility)
    X_train = np.pad(X_train, ((0,0), (2,2), (2,2)), mode='constant') / 255.0
    X_test = np.pad(X_test, ((0,0), (2,2), (2,2)), mode='constant') / 255.0
    
    # Add channel dimension
    X_train = X_train.reshape(-1, 1, 32, 32)
    X_test = X_test.reshape(-1, 1, 32, 32)
    
    # One-hot encode labels
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    return X_train, y_train, X_test, y_test

def load_mnist_data_keras():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train[:1000]  
    y_train = y_train[:1000]
    X_test = X_test[:200]     
    y_test = y_test[:200]
    
    # Normalisasi dan reshape untuk Keras (channels_last format)
    X_train = np.pad(X_train, ((0,0), (2,2), (2,2)), mode='constant') / 255.0
    X_test = np.pad(X_test, ((0,0), (2,2), (2,2)), mode='constant') / 255.0
    
    # Add channel dimension untuk Keras (batch, height, width, channels)
    X_train = X_train.reshape(-1, 32, 32, 1)
    X_test = X_test.reshape(-1, 32, 32, 1)
    
    # One-hot encode labels
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    return X_train, y_train, X_test, y_test


def train_and_validate_model(model, X_train, y_train, X_test, y_test, model_name):
    print(f"\n=== Training {model_name} ===")
    
    # Set loss and optimizer
    model.set(loss=CrossEntropyLoss, optimizer=Adam(learning_rate=0.001), 
              regularization='l2', lambda_param=0.0001)
    
    # Split training data for validation
    split_idx = int(0.8 * len(X_train))
    X_train_split, X_val_split = X_train[:split_idx], X_train[split_idx:]
    y_train_split, y_val_split = y_train[:split_idx], y_train[split_idx:]

    # Train the model
    train_loss, val_loss = model.train(
        X_train_split, y_train_split, 
        epochs=10, 
        batch_size=128,
        X_val=X_val_split, 
        y_val=y_val_split,
        verbose=True
    )
    
    test_pred = model.forward(X_test)
    test_accuracy = np.mean(np.argmax(test_pred, axis=1) == np.argmax(y_test, axis=1))
    
    return train_loss, val_loss, test_accuracy

def train_and_validate_keras_model(model, X_train, y_train, X_test, y_test, model_name):
    print(f"\n=== Training {model_name} (Keras) ===")
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Split training data for validation
    split_idx = int(0.8 * len(X_train))
    X_train_split, X_val_split = X_train[:split_idx], X_train[split_idx:]
    y_train_split, y_val_split = y_train[:split_idx], y_train[split_idx:]
    
    # Train model
    history = model.fit(
        X_train_split, y_train_split,
        batch_size=128,
        epochs=10,
        validation_data=(X_val_split, y_val_split),
        verbose=1 
    )
    
    test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    return history.history['loss'], history.history['val_loss'], test_accuracy