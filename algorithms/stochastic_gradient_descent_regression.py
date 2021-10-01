import numpy as np


# We define our Train Loss fuction (1/2)*np.transpose(X*w-y)*(X*w-y)
def train_loss(d_train, w):
    # We create the matrix X and the matriz y
    X = []
    y = []
    for data in d_train:
        X.append([1] + data[0]), y.append(data[1])
    X = np.array(X)
    y = np.array(y)
    w = np.array(w)
    return (1/2)*np.dot(np.transpose((np.dot(X, w) - y)), (np.dot(X, w) - y))


# We define our Train Loss Gradient fuction (ϕ(x) = x)
def train_loss_gradient(x, y, w):
    return 2 * np.dot((np.dot(w, x) - y), x)


# We define our Gradient descent algorithm
def stochastic_gradient_descent(d_train, w, n, iterations):
    print("Running stochastic gradient descent algorithm")
    iteration = 1
    j = 0
    while iteration <= iterations:
        y = d_train[j][1]
        # We add 1 as the first component in x
        x = np.array([1] + d_train[j][0])
        # We calculate the Train Loss Gradient
        train_loss_gradient_ = train_loss_gradient(x, y, w)
        # We update the vector w
        w = w - n * train_loss_gradient_
        # We calculate the Train Loss
        train_loss_ = train_loss(d_train, w)
        # We print the data
        print(f'In the iteration {iteration}, we get '
              f'w = {w} and TrainLoss(x, y, w) = {train_loss_}')
        iteration += 1
        if j == len(d_train) - 1:
            j = 0
        else:
            j += 1

    print('Algoritm finished')


# We define our DTrain, our weight vector, our n and our iteration number
d_train = [[[0, 0], 0], [[-1, 0], 1 / 2], [[1, 0], 1 / 2], [[0, 1], 1], [[1, 1], 3 / 2]]
w = np.array([0, 1, 1])
n = 0.5
iterations = 5

# We run our stochastic gradient descent algorithm
stochastic_gradient_descent(d_train, w, n, iterations)
