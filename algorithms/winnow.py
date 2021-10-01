import numpy as np


# We define our fw(e) function
def fw(e, w, threshold):
    e = np.array(e)
    w = np.array(w)
    we = np.dot(w, e)
    if we > threshold:
        return 1
    else:
        return 0


# We define our winnow algorithm
def winnow(d_train, alpha):
    # i. We create the w vector and initialize it with weights in 1
    w = []
    e1 = d_train[0][0]
    for _ in e1:
        w.append(1)
    # ii. We create the threshold
    threshold = len(w) - 0.1
    # iii. We start with one training example
    answer = False
    j = 0
    while not answer:
        e = d_train[j][0]
        y = d_train[j][1]
        y_hat = fw(e, w, threshold)
        # iv. If y != y_hat, we update the vector w
        if y != y_hat:
            i = 0
            while i < len(w):
                if e[i] == 1:
                    w[i] *= alpha ** (y - y_hat)
                i += 1
        print(f'\nNow, w = {w}')
        # v. We calculate y_hat for all training examples
        good = True
        for data in d_train:
            e_ = data[0]
            y_ = data[1]
            y_hat_ = fw(e_, w, threshold)
            if y_ != y_hat_:
                good = False
            # We print the data
            i = d_train.index(data) + 1
            print(f'\nPara e{i} = {e_}, y{i} = {y_}')
            print(f'y_hat = {y_hat_}')
            print('Se cumple' if good else 'No se cumple', ' y = y_hat')
        answer = good
        if j == len(d_train) - 1:
            j = 0
        else:
            j += 1
    print('\nAlgoritm finished')


# We define our DTrain and our alpha
"""
d_train = [[[1, 1, 0, 1, 0], 1], [[0, 1, 1, 0, 0], 0], [[1, 1, 0, 0, 1], 1],
           [[0, 1, 1, 1, 0], 0], [[1, 1, 1, 1, 1], 0]]
"""

d_train = [[[1, 1, 0, 1, 0], 1], [[0, 1, 1, 0, 0], 0], [[1, 1, 0, 0, 1], 1],
           [[1, 0, 1, 1, 0], 0], [[0, 1, 0, 1, 0], 0]]
alpha = 2

# We run our winnow algorithm
winnow(d_train, alpha)
