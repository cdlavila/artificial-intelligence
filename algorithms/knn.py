import numpy as np
import math
import statistics as st


# We define our function of norm of a vector
def norm(v):
    return np.linalg.norm(np.array(v))


# We define our fuction of angle between two vectors
def angle(v1, v2):
    dot_product = np.dot(np.array(v1), np.array(v2))
    division = dot_product / (norm(v1) * norm(v2))
    return math.acos(division)


# We define our K-NN algorithm
def knn(d_train, k, v):
    print("All neighbors")
    # We calculate the angles between v and all vector from D_Train and sort them ascendingly
    angles = []
    for data in d_train:
        element = [angle(data[0], v), data[0], data[1]]
        angles.append(element)
        print(element)
    angles.sort()
    # Here, we have the k nearest neighbors and their votes
    nearest_neighbors = angles[:k]
    # Here, we have the votes from the k nearest neighbors
    votes = []
    for n in nearest_neighbors:
        votes.append(n[2])
    # We print the Nearest neihbors
    print('Nearest neighbors')
    for neighbor in nearest_neighbors:
        print(neighbor)
    # We return the most repeated vote
    return st.mode(votes)


# We define our DTrain, our k and our vector to clasific
d_train = [[[5, 2, 0, 0, 3, 2, 1, 0, 0, 3, 0, 3], 'A'], [[3, 1, 0, 1, 0, 2, 0, 1, 0, 4, 1, 2], 'A'],
           [[0, 2, 3, 0, 0, 0, 2, 3, 2, 0, 1, 0], 'B'], [[3, 1, 4, 0, 0, 0, 3, 2, 3, 2, 0, 4], 'B'],
           [[0, 3, 4, 0, 0, 0, 3, 2, 0, 0, 2, 0], 'B'], [[3, 2, 3, 0, 0, 2, 2, 0, 0, 3, 0, 4], 'A']]
k = 3
v = [5, 1, 0, 1, 1, 0, 2, 2, 0, 3, 0, 2]

# We run our K-NN algorithm
group_of_v = knn(d_train, k, v)
print(f'The vector {v} belong to class {group_of_v}')
