import numpy as np

class Activation:
    def forward(self, inputs):
        raise NotImplementedError
        
    def backward(self, grad_outputs):
        raise NotImplementedError

class Linear(Activation):
    def forward(self, inputs):
        self.inputs = inputs
        return inputs
        
    def backward(self, grad_outputs):
        return grad_outputs

class Sigmoid(Activation):
    def forward(self, inputs):
        self.output = 1 / (1 + np.exp(-inputs))
        return self.output
        
    def backward(self, grad_outputs):
        return grad_outputs * self.output * (1 - self.output)

class ReLU(Activation):
    def forward(self, inputs):
        self.inputs = inputs
        return np.maximum(0, inputs)
        
    def backward(self, grad_outputs):
        grad = np.copy(grad_outputs)
        grad[self.inputs <= 0] = 0
        return grad

class Tanh(Activation):
    def forward(self, inputs):
        self.output = np.tanh(inputs)
        return self.output
        
    def backward(self, grad_outputs):
        return grad_outputs * (1 - self.output**2)

class LeakyReLU(Activation):
    def __init__(self, alpha=0.01):
        self.alpha = alpha
        
    def forward(self, inputs):
        self.inputs = inputs
        return np.where(inputs > 0, inputs, self.alpha * inputs)
        
    def backward(self, grad_outputs):
        grad = np.ones_like(self.inputs)
        grad[self.inputs < 0] = self.alpha
        return grad * grad_outputs

class Softmax(Activation):
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self.output
        
    def backward(self, grad_outputs):
        return grad_outputs

class DenseLayer:
    def __init__(self, n_inputs, n_neurons, activation, weight_initializer='random'):
        if weight_initializer == 'he':
            std = np.sqrt(2.0 / n_inputs)
            self.weights = np.random.normal(0, std, (n_inputs, n_neurons))
        elif weight_initializer == 'xavier':
            std = np.sqrt(1.0 / n_inputs)
            self.weights = np.random.normal(0, std, (n_inputs, n_neurons))
        else:
            self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
            
        self.biases = np.zeros((1, n_neurons))
        self.activation = activation() if isinstance(activation, type) else activation
        self.regularization = None
        self.lambda_param = 0.01

    def forward(self, inputs):
        if len(inputs.shape) == 1:
            inputs = inputs.reshape(1, -1)
        self.inputs = inputs
        self.z = np.dot(inputs, self.weights) + self.biases
        self.output = self.activation.forward(self.z)
        return self.output

    def backward(self, grad_outputs):
        grad_activation = self.activation.backward(grad_outputs)
        
        self.dweights = np.dot(self.inputs.T, grad_activation)
        self.dbiases = np.sum(grad_activation, axis=0, keepdims=True)
        
        if self.regularization == 'l2':
            self.dweights += self.lambda_param * self.weights
        elif self.regularization == 'l1':
            self.dweights += self.lambda_param * np.sign(self.weights)
            
        self.dinputs = np.dot(grad_activation, self.weights.T)
        return self.dinputs

class Loss:
    def calculate(self, y_pred, y_true):
        raise NotImplementedError
        
    def backward(self, y_pred, y_true):
        raise NotImplementedError

class MeanSquaredError(Loss):
    def calculate(self, y_pred, y_true):
        return np.mean((y_pred - y_true)**2)
        
    def backward(self, y_pred, y_true):
        n_samples = len(y_true)
        return 2 * (y_pred - y_true) / n_samples

class CrossEntropyLoss(Loss):
    def calculate(self, y_pred, y_true):
        n_samples = len(y_true)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(n_samples), y_true]
        else:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        
        return -np.mean(np.log(correct_confidences))
    
    def backward(self, y_pred, y_true):
        n_samples = len(y_true)
        n_classes = y_pred.shape[1]

        if len(y_true.shape) == 1:
            y_true_one_hot = np.zeros((n_samples, n_classes))
            y_true_one_hot[np.arange(n_samples), y_true] = 1
        else:
            y_true_one_hot = y_true

        return (y_pred - y_true_one_hot) / n_samples

class Optimizer:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        
    def update_params(self, layer):
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, learning_rate=0.01, decay=0.0):
        super().__init__(learning_rate)
        self.decay = decay
        self.iterations = 0
        self.current_lr = learning_rate
        
    def update_params(self, layer):
        if self.decay:
            self.current_lr = self.learning_rate * (1.0 / (1.0 + self.decay * self.iterations))
            
        layer.weights -= self.current_lr * layer.dweights
        layer.biases -= self.current_lr * layer.dbiases
        self.iterations += 1

class Adagrad(Optimizer):
    def __init__(self, learning_rate=0.01, epsilon=1e-7):
        super().__init__(learning_rate)
        self.epsilon = epsilon
        self.cache = {}
        
    def update_params(self, layer):
        if id(layer) not in self.cache:
            self.cache[id(layer)] = {'weights': np.zeros_like(layer.weights), 
                                    'biases': np.zeros_like(layer.biases)}
            
        self.cache[id(layer)]['weights'] += layer.dweights**2
        self.cache[id(layer)]['biases'] += layer.dbiases**2
        
        layer.weights -= self.learning_rate * layer.dweights / (np.sqrt(self.cache[id(layer)]['weights']) + self.epsilon)
        layer.biases -= self.learning_rate * layer.dbiases / (np.sqrt(self.cache[id(layer)]['biases']) + self.epsilon)

class Adam(Optimizer): 
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-7):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0 

    def update_params(self, layer):
        self.t += 1
        layer_id = id(layer)
        
        if layer_id not in self.m:
            self.m[layer_id] = {'weights': np.zeros_like(layer.weights), 
                                'biases': np.zeros_like(layer.biases)}
            self.v[layer_id] = {'weights': np.zeros_like(layer.weights), 
                                'biases': np.zeros_like(layer.biases)}
        
        self.m[layer_id]['weights'] = self.beta1 * self.m[layer_id]['weights'] + (1 - self.beta1) * layer.dweights
        self.m[layer_id]['biases'] = self.beta1 * self.m[layer_id]['biases'] + (1 - self.beta1) * layer.dbiases
        
        m_corrected_w = self.m[layer_id]['weights'] / (1 - self.beta1**self.t)
        m_corrected_b = self.m[layer_id]['biases'] / (1 - self.beta1**self.t)

        self.v[layer_id]['weights'] = self.beta2 * self.v[layer_id]['weights'] + (1 - self.beta2) * (layer.dweights**2)
        self.v[layer_id]['biases'] = self.beta2 * self.v[layer_id]['biases'] + (1 - self.beta2) * (layer.dbiases**2)

        v_corrected_w = self.v[layer_id]['weights'] / (1 - self.beta2**self.t)
        v_corrected_b = self.v[layer_id]['biases'] / (1 - self.beta2**self.t)

        layer.weights -= self.learning_rate * m_corrected_w / (np.sqrt(v_corrected_w) + self.epsilon)
        layer.biases -= self.learning_rate * m_corrected_b / (np.sqrt(v_corrected_b) + self.epsilon)

class myANN:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.optimizer = None
        self.regularization = None
        self.lambda_param = 0.01

    def add(self, layer):
        self.layers.append(layer)

    def set(self, loss, optimizer, regularization=None, lambda_param=0.01):
        self.loss = loss()
        self.optimizer = optimizer
        self.regularization = regularization
        self.lambda_param = lambda_param
        
        for layer in self.layers:
            layer.regularization = regularization
            layer.lambda_param = lambda_param

    def forward(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, y_pred, y_true):
        grad = self.loss.backward(y_pred, y_true)
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def update(self):
        self.optimizer.update_params(self.layers[-1])
        for layer in reversed(self.layers[:-1]):
            self.optimizer.update_params(layer)

    def train(self, X, y, epochs, batch_size, X_val=None, y_val=None, verbose=True):
        train_loss_history = []
        val_loss_history = []
        n_samples = len(X)
        
        for epoch in range(epochs):
            permutation = np.random.permutation(n_samples)
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]
            
            epoch_loss = 0
            
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                y_pred = self.forward(X_batch)
                loss = self.loss.calculate(y_pred, y_batch)
                
                reg_loss = 0
                if self.regularization == 'l2':
                    for layer in self.layers:
                        reg_loss += 0.5 * self.lambda_param * np.sum(layer.weights * layer.weights)
                elif self.regularization == 'l1':
                    for layer in self.layers:
                        reg_loss += self.lambda_param * np.sum(np.abs(layer.weights))
                
                total_loss = loss + reg_loss
                epoch_loss += total_loss
                
                self.backward(y_pred, y_batch)
                self.update()
            
            avg_loss = epoch_loss / (n_samples / batch_size)
            train_loss_history.append(avg_loss)
            
            if X_val is not None and y_val is not None:
                val_pred = self.forward(X_val)
                val_loss = self.loss.calculate(val_pred, y_val)
                val_loss_history.append(val_loss)
            
            if verbose:
                val_info = f", Val Loss: {val_loss:.4f}" if X_val is not None else ""
                print(f'Epoch {epoch}, Train Loss: {avg_loss:.4f}{val_info}')
        
        return train_loss_history, val_loss_history

    def predict(self, X):
        y_pred = self.forward(X)
        if hasattr(self.layers[-1].activation, 'output'):
            if self.layers[-1].activation.output.shape[1] > 1:
                return np.argmax(y_pred, axis=1)
        return y_pred