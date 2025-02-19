import numpy as np
from keras.datasets import cifar10

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

num_classes = 10
def one_hot_encode(labels, num_classes):
    encoded = np.zeros((labels.size, num_classes))
    encoded[np.arange(labels.size), labels.flatten()] = 1
    return encoded

train_labels = one_hot_encode(train_labels, num_classes)
test_labels = one_hot_encode(test_labels, num_classes)

def initialize_filters(shape):
    return np.random.randn(*shape) * 0.1

def batch_norm(x):
    mean = np.mean(x, axis=0)
    var = np.var(x, axis=0)
    return (x - mean) / np.sqrt(var + 1e-9)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=-1, keepdims=True)

def conv2d(image, filters):
    num_filters, filter_height, filter_width, _ = filters.shape
    height, width, channels = image.shape
    output = np.zeros((height - filter_height + 1, width - filter_width + 1, num_filters))
    for f in range(num_filters):
        for i in range(height - filter_height + 1):
            for j in range(width - filter_width + 1):
                region = image[i:i+filter_height, j:j+filter_width, :]
                output[i, j, f] = np.sum(region * filters[f])
    return output

def max_pooling(feature_map, pool_size=(2, 2)):
    height, width, num_filters = feature_map.shape
    pooled_height = height // pool_size[0]
    pooled_width = width // pool_size[1]
    pooled = np.zeros((pooled_height, pooled_width, num_filters))
    for f in range(num_filters):
        for i in range(0, pooled_height * pool_size[0], pool_size[0]):
            for j in range(0, pooled_width * pool_size[1], pool_size[1]):
                region = feature_map[i:i+pool_size[0], j:j+pool_size[1], f]
                pooled[i//2, j//2, f] = np.max(region)
    return pooled

def flatten(feature_map):
    return feature_map.flatten()

def cross_entropy_loss(predictions, labels):
    predictions = np.clip(predictions, 1e-9, 1 - 1e-9)
    return -np.sum(labels * np.log(predictions))

def grad_cross_entropy(predictions, labels):
    return predictions - labels

filters1 = initialize_filters((16, 3, 3, 3))
filters2 = initialize_filters((32, 3, 3, 16))
flattened_size = ((32 - 6 + 1) // 4) ** 2 * 32
dense_weights = np.random.randn(flattened_size, num_classes) * 0.1
dense_bias = np.random.randn(num_classes) * 0.1
learning_rate = 0.001
batch_size = 32

def forward_pass(image):
    conv_out1 = conv2d(image, filters1)
    conv_out_activated1 = relu(batch_norm(conv_out1))
    pooled_out1 = max_pooling(conv_out_activated1)

    conv_out2 = conv2d(pooled_out1, filters2)
    conv_out_activated2 = relu(batch_norm(conv_out2))
    pooled_out2 = max_pooling(conv_out_activated2)

    flattened = flatten(pooled_out2)
    dense_out = np.dot(flattened, dense_weights) + dense_bias
    output = softmax(dense_out)
    return conv_out1, conv_out_activated1, pooled_out1, conv_out2, conv_out_activated2, pooled_out2, flattened, dense_out, output

def backward_pass(image, conv_out1, conv_out_activated1, pooled_out1, conv_out2, conv_out_activated2, pooled_out2, flattened, dense_out, output, label):
    global filters1, filters2, dense_weights, dense_bias
    
    loss_grad = grad_cross_entropy(output, label)
    dense_weights_grad = np.outer(flattened, loss_grad)
    dense_bias_grad = loss_grad

    pooled_grad2 = loss_grad @ dense_weights.T
    pooled_grad2 = pooled_grad2.reshape(pooled_out2.shape)

    expanded_pooled_grad2 = np.zeros(conv_out_activated2.shape)
    for i in range(pooled_grad2.shape[0]):
        for j in range(pooled_grad2.shape[1]):
            expanded_pooled_grad2[i, j] = pooled_grad2[i//2, j//2]

    conv_grad2 = expanded_pooled_grad2 * relu_derivative(conv_out_activated2)

    filter_gradients2 = np.zeros_like(filters2)
    for f in range(filters2.shape[0]):
        for i in range(conv_grad2.shape[0]):
            for j in range(conv_grad2.shape[1]):
                region = pooled_out1[i:i+3, j:j+3, :]
                filter_gradients2[f] += conv_grad2[i, j, f] * region
                
    dense_weights -= learning_rate * dense_weights_grad
    dense_bias -= learning_rate * dense_bias_grad
    filters2 -= learning_rate * filter_gradients2

def train_cnn(train_images, train_labels, epochs=5):
    for epoch in range(epochs):
        correct = 0
        total_loss = 0
        for i in range(0, len(train_images), batch_size):
            batch_images = train_images[i:i+batch_size]
            batch_labels = train_labels[i:i+batch_size]
            for j in range(len(batch_images)):
                outputs = forward_pass(batch_images[j])
                total_loss += cross_entropy_loss(outputs[-1], batch_labels[j])
                backward_pass(batch_images[j], *outputs, batch_labels[j])
                if np.argmax(outputs[-1]) == np.argmax(batch_labels[j]):
                    correct += 1
        print(f"Epoch {epoch+1}, Accuracy: {correct / len(train_images):.4f}, Loss: {total_loss / len(train_images):.4f}")

def test_cnn(test_images, test_labels):
    correct = 0
    for i in range(len(test_images)):
        _, _, _, _, _, _, _, _, output = forward_pass(test_images[i])
        if np.argmax(output) == np.argmax(test_labels[i]):
            correct += 1
    print(f"Test Accuracy: {correct / len(test_images):.4f}")

train_cnn(train_images[:10000], train_labels[:10000], epochs=10)
test_cnn(test_images, test_labels)
