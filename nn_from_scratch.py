# %% 1 
# Package imports 
import matplotlib.pyplot as plt 
import numpy as np 
import sklearn 
import sklearn.datasets 
import sklearn.linear_model 
import matplotlib 
import loadmnist as mnist
 
# Display plots inline and change default figure size 
# %matplotlib inline 
matplotlib.rcParams['figure.figsize'] = (10.0, 8.0) 
 
# %% 2 
np.random.seed(3) 
X, y = sklearn.datasets.make_moons(200, noise=0.20) 
plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral) 

train_labels, train_images = mnist.read(dataset="training")
test_labels, test_images = mnist.read(dataset="testing")

X, y = train_images, train_labels

nsamples, nx, ny = train_images.shape
X = train_images.reshape((nsamples,nx*ny))

nsamples, nx, ny = test_images.shape
test_images = test_images.reshape((nsamples, nx*ny))


class NeuralNet:
    def __init__(self, output_dim, eps, lmbda, nn_hdim=3):
        # self.X = X_
        # self.y = y_
        # self.test_images = test_x
        # self.test_labels = test_y
        # self.num_examples = X.shape[0] # training set size 
        # self.nn_input_dim = X.shape[1] # input layer dimensionality 
        self.nn_output_dim = output_dim # output layer dimensionality 
        self.nn_hdim = nn_hdim 
        # Gradient descent parameters (I picked these by hand) 
        self.epsilon = eps# 0.000001 # learning rate for gradient descent 
        self.reg_lambda = lmbda#0.01 # regularization strength
        

    # %% 7 
    # Helper function to evaluate the total loss on the dataset 
    def calculate_loss(self, X, y): 
        W1, b1, W2, b2 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2'] 
        # Forward propagation to calculate our predictions 
        reg_lambda = self.reg_lambda
        num_examples = X.shape[0]

        z1 = X.dot(W1) + b1 
        a1 = np.tanh(z1) 
        z2 = a1.dot(W2) + b2 
        exp_scores = np.exp(z2) 
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) 
        # Calculating the loss

        # was previously correct_logprobs = -np.log(probs[range(num_examples), y])

        #loops through each index and looks for the probability predicted for the right answer 
        correct_logprobs = -np.log(probs[range(num_examples), y])
        # print("correct log probs is ", correct_logprobs)
        data_loss = np.sum(correct_logprobs) 
        # Add regulatization term to loss (optional) 

        data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2))) 
        return 1./num_examples * data_loss 
     
    # %% 8 
    # Helper function to predict an output (0 or 1) 
    def predict(self, x): 
        W1, b1, W2, b2 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2'] 
        # Forward propagation 
        z1 = x.dot(W1) + b1 
        a1 = np.tanh(z1) 
        z2 = a1.dot(W2) + b2 
        exp_scores = np.exp(z2) 
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) 
        return np.argmax(probs, axis=1) 

    def test(self, test_images, test_labels):
        num_correct = 0.
        total_num = test_images.shape[0]
        for i, image in enumerate(test_images):
            guess = self.predict(image)
            actual = test_labels[i]
            # if guess == actual:
            #     print("guess was ", guess, " and actual was ", actual)
            num_correct += guess == actual
        return float(num_correct)/total_num*100 
     
    # %% 16 
    # This function learns parameters for the neural network and returns the model. 
    # - nn_hdim: Number of nodes in the hidden layer 
    # - num_passes: Number of passes through the training data for gradient descent 
    # - print_loss: If True, print the loss every 1000 iterations 
    def build_model(self, X, y, num_passes=20000, print_loss=False): 

        # Gradient descent. For each batch...
        self.nn_input_dim = X.shape[1]

        np.random.seed(0) 
        W1 = np.random.randn(self.nn_input_dim, self.nn_hdim) / np.sqrt(self.nn_input_dim) 
        b1 = np.zeros((1, self.nn_hdim)) 
        W2 = np.random.randn(self.nn_hdim, self.nn_output_dim) / np.sqrt(self.nn_hdim) 
        b2 = np.zeros((1, self.nn_output_dim))

        self.model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2} 
        W1, b1, W2, b2 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2']
        nn_hdim = self.nn_hdim
        reg_lambda = self.reg_lambda
        epsilon = self.epsilon
        num_examples = X.shape[0]

        for i in range(0, num_passes): 
            
            # Forward propagation 
            z1 = X.dot(W1) + b1 
            a1 = np.tanh(z1) 
            z2 = a1.dot(W2) + b2 
            exp_scores = np.exp(z2) 
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) 
     
            # Backpropagation 
            delta3 = probs 
            delta3[range(num_examples), y] -= 1 
            dW2 = (a1.T).dot(delta3) 
            db2 = np.sum(delta3, axis=0, keepdims=True) 
            delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2)) 
            dW1 = np.dot(X.T, delta2) 
            db1 = np.sum(delta2, axis=0) 
     
            # Add regularization terms (b1 and b2 don't have regularization terms) 
            dW2 += reg_lambda * W2 
            dW1 += reg_lambda * W1 
     
            # Gradient descent parameter update 
            W1 += -epsilon * dW1 
            b1 += -epsilon * db1 
            W2 += -epsilon * dW2 
            b2 += -epsilon * db2 
     
            # Assign new parameters to the model 
            self.model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2} 
     
            # Optionally print the loss. 
            # This is expensive because it uses the whole dataset, so we don't want to do it too often. 
            if print_loss and i % 500 == 0: 
              print("Loss after iteration %i: %f" %(i, self.calculate_loss(X, y)))
            if i % 100 == 0:
                print("Completed iteration ",i)

        print("Loss after iteration %i: %f" %(num_passes, self.calculate_loss(X, y)))
        return self.model 
 
# %% 17 
# Build a model with a 3-dimensional hidden layer
class AdaBoostNNs:
    def __init__(self, num_nns, num_passes):
        # num_per_nn = train_x.shape[0]/num_nns
        self.nns = []
        self.n = num_nns
        self.num_passes = num_passes

        for t in range(num_nns):
            # X = train_x
            # y = train_y
            # test_images = test_x
            # test_labels = test_y
            output_dim = 10
            eps = 0.000001
            lmbda = 0.01
            net = NeuralNet(output_dim, eps, lmbda)
            self.nns.append(net)

    # Adapted from the Hw12 solution code 
    def fit(self, X, y):
        self.w = np.ones(X.shape[0]) / X.shape[0]  # Weights on data
        self.a = np.zeros(self.n)  # Weights on decision trees
        i = 0

        while i < self.n:
            idx = np.random.choice(X.shape[0], size=X.shape[0], p=self.w)
            newX, newy = X[idx, :], y[idx]

            self.nns[i].build_model(newX, newy, num_passes = self.num_passes)
            wrong = np.abs((y - self.nns[i].predict(X)) != 0)
            error = wrong.dot(self.w) / np.sum(self.w)
            self.a[i] = 0.5 * np.log((1 - error) / error)
            
            # Update w
            wrong_idx = np.where(wrong == 1)[0]
            right_idx = np.where(wrong != 1)[0]
            self.w[wrong_idx] = self.w[wrong_idx] * np.exp(self.a[i])
            self.w[right_idx] = self.w[right_idx] * np.exp(-self.a[i])
            self.w /= np.sum(self.w)
            i += 1
        return self

    def predict(self, X):
        yhat = np.asarray([self.nns[i].predict(X) for i in range(self.n)])
        # print(yhat.shape)

        p0 = self.a.dot(yhat == 0)
        p1 = self.a.dot(yhat == 1)
        p2 = self.a.dot(yhat == 2)
        p3 = self.a.dot(yhat == 3)
        p4 = self.a.dot(yhat == 4)
        p5 = self.a.dot(yhat == 5)
        p6 = self.a.dot(yhat == 6)
        p7 = self.a.dot(yhat == 7)
        p8 = self.a.dot(yhat == 8)
        p9 = self.a.dot(yhat == 9)

        # print(p0.shape)
        raw_guesses = np.asarray(np.vstack([p0, p1, p2, p3, p4, p5, p6, p7, p8, p9]))
        predictions = np.ones(X.shape[0])
        predictions = np.array(np.argmax(np.vstack([p0, p1, p2, p3, p4, p5, p6, p7, p8, p9]), axis=0))
        # num_right = 0
        # for i in range(raw_guesses.shape[1]):
        #     guess = np.argmax(raw_guesses[:,i])
        #     predictions[i] = guess
            # if guess == train_labels[i]:
            #     num_right +=1
        # print("num for sure right is ", num_right)
        # print(predictions.shape)
        return predictions

    def test(self, X, y):
        preds = self.predict(X)

        # look at the number of things where the difference was 0
        # divide by the total number of things to get the number you got right
        num_right = np.sum((preds - y) == 0)
        # print("num_right ", num_right)
        right = float(num_right/y.shape[0])
        # right = 1. - error
        
        return right * 100


nn_1 = NeuralNet(10, 0.000001, 0.01)
nn_1.build_model(X, y, num_passes=100)
print("baseline is ", nn_1.test(test_images, test_labels))

# print("percentage correct is ", nn_1.test())

adaThing = AdaBoostNNs(7, 100)
adaThing.fit(X, y)
print("ada boost correct is ",adaThing.test(test_images, test_labels), "%")


