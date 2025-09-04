from supervised_learning.ANN import myANN, DenseLayer, Conv2DLayer, MaxPooling2DLayer, FlattenLayer, ReLU, Softmax

def create_zfnet_model():
    model = myANN()
    
    model.add(Conv2DLayer(n_filters=96, kernel_size=7, strides=2, activation=ReLU))
    model.add(MaxPooling2DLayer(pool_size=3, strides=2))
    
    model.add(Conv2DLayer(n_filters=256, kernel_size=5, strides=2, activation=ReLU))
    model.add(MaxPooling2DLayer(pool_size=3, strides=2))
    
    model.add(Conv2DLayer(n_filters=512, kernel_size=3, use_padding=True, activation=ReLU))
    model.add(Conv2DLayer(n_filters=1024, kernel_size=3, use_padding=True, activation=ReLU))
    model.add(Conv2DLayer(n_filters=512, kernel_size=3, use_padding=True, activation=ReLU))
    model.add(MaxPooling2DLayer(pool_size=3, strides=2))
    
    model.add(FlattenLayer())
    model.add(DenseLayer(n_inputs=512 * 6 * 6, n_neurons=4096, activation=ReLU))
    model.add(DenseLayer(n_inputs=4096, n_neurons=4096, activation=ReLU))
    model.add(DenseLayer(n_inputs=4096, n_neurons=10, activation=Softmax)) # Asumsi 10 kelas

    return model