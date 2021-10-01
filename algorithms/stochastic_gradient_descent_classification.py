import numpy as np


# We define our Hinge Train Loss fuction
def hinge_train_loss(d_train, w):
    summation = 0
    for data in d_train:
        x = np.array([1] + data[0])
        y = data[1]
        summation += max(1-np.dot(w, x)*y, 0)
    return summation


# We define our Hinge Train Loss Gradient fuction (ϕ(x) = x)
def hinge_train_loss_gradient(x, y, w):
    # We add 1 as the first component in x
    x = np.array([1] + x)
    return -1*(np.dot(w, x)*y < 1)*x*y


# We define our Gradient descent algorithm
def stochastic_gradient_descent(d_train, w, n, iterations):
    print("Running stochastic gradient descent algorithm")
    iteration = 1
    j = 0
    while iteration <= iterations:
        y = d_train[j][1]
        # We add 1 as the first component in x
        x = d_train[j][0]
        # We calculate the Hinge Train Loss Gradient
        hinge_train_loss_gradient_ = hinge_train_loss_gradient(x, y, w)
        # We update the vector w
        w = w - n * hinge_train_loss_gradient_
        # We calculate the Hinge Train Loss
        hinge_train_loss_ = hinge_train_loss(d_train, w)
        # We print the data
        print(f'In the iteration {iteration}, we get '
              f'w = {w} and TrainLossHinge(x, y, w) = {hinge_train_loss_}')
        iteration += 1
        if j == len(d_train) - 1:
            j = 0
        else:
            j += 1

    print('Algoritm finished')


# We define our DTrain, our weight vector, our n and our iteration number
d_train = [[[0, 0], 1], [[-1, 0], -1], [[1, 0], 1], [[0, 1], -1], [[1, 1], 1]]
w = np.array([0, 1, 1])
n = 0.5
iterations = 41

# We run our stochastic gradient descent algorithm
stochastic_gradient_descent(d_train, w, n, iterations)
