import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.python.framework import ops
import h5py


def load_dataset():
    images = np.load('0-9 data/X.npy')
    labels = np.load('0-9 data/Y.npy')
    np.random.seed(3)
    permutation = list (np.random.permutation (images.shape[0]))
    images = images[permutation , : , :]
    labels = labels[permutation , :]

    train_X = images[:1856]
    test_X = images[1856:]

    train_Y = labels[:1856]
    test_Y = labels[1856:]

    zero = np.zeros((1856, 64, 64))
    zero1 = np.zeros((206, 64, 64))

    train_X = np.stack ((train_X, zero, zero)  , -1)
    test_X = np.stack ((test_X, zero1, zero1)  , -1)

    return train_X, test_X, train_Y, test_Y


def make_mini_batches(X, Y, mini_batch_size, seed):
    m = X.shape[0]
    mini_batches = []
    np.random.seed(seed)

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :, :, :]
    shuffled_Y = Y[permutation, :]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = int(m/mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size: m, :,:, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def initialize_parameters():
    tf.set_random_seed(0)
    parameters = {}
    parameters["W1"] = tf.get_variable('W1', [4, 4, 3, 8], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    parameters["W2"] = tf.get_variable('W2', [4, 4, 8, 16], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    parameters["W3"] = tf.get_variable('W3', [2, 2, 16, 32], initializer=tf.contrib.layers.xavier_initializer(seed=0))


    return parameters


def forward_prop(X , parameters):
    Z1 = tf.nn.conv2d (X , parameters["W1"] , strides=[1 , 1 , 1 , 1] , padding='SAME')
    A1 = tf.nn.relu (Z1)
    P1 = tf.nn.max_pool (A1 , ksize=[1 , 8 , 8 , 1] , strides=[1 , 8 , 8 , 1] , padding="SAME")

    Z2 = tf.nn.conv2d (P1 , parameters["W2"] , strides=[1 , 1 , 1 , 1] , padding="SAME")
    A2 = tf.nn.relu (Z2)
    P2 = tf.nn.max_pool (A2 , ksize=[1 , 4 , 4 , 1] , strides=[1 , 4 , 4 , 1] , padding="SAME")

    Z3 = tf.nn.conv2d (P2 , parameters["W3"] , strides=[1 , 1 , 1 , 1] , padding="SAME")
    A3 = tf.nn.relu (Z3)
    P3 = tf.nn.max_pool (A3 , ksize=[1 , 2 , 2 , 1] , strides=[1 , 2 , 2 , 1] , padding="SAME")

    flatten = tf.contrib.layers.flatten (P3)
    FC = tf.contrib.layers.fully_connected (flatten , 10 , activation_fn=None)

    return FC


def compute_cost(FC, Y, para):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=FC, labels=Y)+ .01 *tf.nn.l2_loss(para["W1"])+.01 * tf.nn.l2_loss(para["W2"])
                          + .01*tf.nn.l2_loss(para["W3"]))
    return cost


def model(X_train, X_test, Y_train, Y_test, learning_rate=.008, epochs=100, minibatch_size=64, print_cost=True):
    tf.set_random_seed(0)
    costs = []
    seed = 3
    ops.reset_default_graph()

    X = tf.placeholder(tf.float32, shape=[None, 64, 64, 3], name="Placeholder")
    Y = tf.placeholder(tf.float32, shape=[None, 10], name="Placeholder")
    m = X_train.shape[0]
    parameters = initialize_parameters()
    FC = forward_prop(X, parameters)
    cost = compute_cost(FC, Y, parameters)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.global_variables_initializer()
    with tf.Session()as sess:
        sess.run(init)
        for epoch in range(epochs):
            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size)
            seed = seed + 1
            minibatches = makemini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                _, temp_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})

                minibatch_cost += temp_cost / num_minibatches

            if print_cost == True and epoch % 5 == 0:
                print("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if epoch % 1 == 0:
                costs.append(minibatch_cost)
        #Saving best weights
        np.save ("W1.npy" , parameters["W1"].eval (session=sess))
        np.save ("W2.npy" , parameters["W2"].eval (session=sess))
        np.save ('W3.npy' , parameters["W3"].eval (session=sess))

        #Plotting
        plt.plot(np.squeeze(costs))
        plt.ylabel ('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        predict_op = tf.argmax(FC, 1)
        correct_prediction = tf.equal (predict_op , tf.argmax (Y , 1))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean (tf.cast (correct_prediction , "float"))
        print (accuracy)
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        print ("Train Accuracy:", train_accuracy)
        print ("Test Accuracy:", test_accuracy)

        return parameters


def predict(parameters, X):
    W1 = tf.convert_to_tensor (parameters["W1"])
    W2 = tf.convert_to_tensor (parameters["W2"])
    W3 = tf.convert_to_tensor (parameters["W3"])

    params = {"W1": W1 ,
              "W2": W2 ,
              "W3": W3 ,
              }
    x = tf.placeholder ("float" , [1, 64, 64, 3])

    z3 = forward_prop (x , params)
    p = tf.argmax (z3)

    sess = tf.Session ()
    prediction = sess.run (p , feed_dict={x: X})

    return prediction


if __name__ == '__main__':
    parameters = {}
    X_train , X_test , Y_train , Y_test = load_dataset ()
    W1 , W2 , W3 = model (X_train , X_test , Y_train , Y_test , learning_rate=.009 , epochs=150 , minibatch_size=64)

   
