import os
import numpy
import random
from skimage import io, transform, color
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.compat.v1 as v1

def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f)
                      for f in os.listdir(label_directory)
                      if f.endswith(".ppm")]
        for f in file_names:
            images.append(io.imread(f))
            labels.append(int(d))
    return images, labels

ROOT_PATH = os.getcwd()
train_data_directory = os.path.join(ROOT_PATH, 'traffic_images/Training')
test_data_directory = os.path.join(ROOT_PATH, 'traffic_images/Testing')

images, labels = load_data(train_data_directory)
images28 = color.rgb2gray(numpy.array([transform.resize(image, (28, 28)) for image in images]))
unique_labels = set(labels)

# for i, label in enumerate(unique_labels):
#     plt.subplot(8, 8, i+1)
#     plt.axis('off')
#     plt.title("Label {0} ({1})".format(label, labels.count(label)))
#     plt.imshow(images28[i])

# traffic_signs = [300, 2250, 3650, 4000]

# for i in range(len(traffic_signs)):
#     plt.subplot(1, 4, i+1)
#     plt.axis('off')
#     plt.imshow(images28[traffic_signs[i]], cmap="gray")
#     plt.subplots_adjust(wspace=0.5)

v1.disable_eager_execution()
x = v1.placeholder(dtype=v1.float32, shape = [None, 28, 28])
y = v1.placeholder(dtype=v1.int32, shape = [None])

images_flat = v1.layers.flatten(x)
logits = v1.layers.dense(images_flat, len(unique_labels))
loss = v1.reduce_mean(v1.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = logits))
train_op = v1.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
correct_pred = v1.argmax(logits, 1)
accuracy = v1.reduce_mean(v1.cast(correct_pred, v1.float32))

print("images_flat: ", images_flat)
print("logits: ", logits)
print("loss: ", loss)
print("predicted_labels: ", correct_pred)

v1.set_random_seed(1234)
sess = v1.Session()
sess.run(v1.global_variables_initializer())
NUM_EPOCHS = 3000

for i in range(NUM_EPOCHS):
    print('EPOCH', i)
    _, accuracy_val = sess.run([train_op, accuracy], feed_dict={x: images28, y: labels})
    if i % 10 == 0:
        print("Loss: ", accuracy_val)
    print('DONE WITH EPOCH', i)

test_images, test_labels = load_data(test_data_directory)
test_images28 = color.rgb2gray(numpy.array([transform.resize(image, (28, 28)) for image in test_images]))


predicted = sess.run([correct_pred], feed_dict={x: test_images28})[0]

match_count = sum([int(y == y_) for y, y_ in zip(test_labels, predicted)])

accuracy_val = match_count / len(test_labels)

print("Accuracy: {:.3f}".format(accuracy_val))

sess.close()