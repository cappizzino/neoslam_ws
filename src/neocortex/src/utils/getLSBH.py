import numpy as np

def get_lsbh(feature, matrix_p, number_s):
    'LSBH'
    numbers_one = int(round(matrix_p.shape[1]*number_s))
    # random projection
    y_2 = np.dot(feature, matrix_p)
    # sort
    i_dx = np.sort(y_2, axis=1)
    i_max = y_2.shape[1] - 1
    l_1 = np.zeros((y_2.shape[0], y_2.shape[1]), dtype=np.bool, order='F')
    l_2 = np.zeros((y_2.shape[0], y_2.shape[1]), dtype=np.bool, order='F')

    for i in range(y_2.shape[0]):
        for j in range(numbers_one):
            indx = np.where((y_2[i, :] == i_dx[i, i_max-j]))
            l_1[i, indx] = 1
            indx = np.where((y_2[i, :] == i_dx[i, j]))
            l_2[i, indx] = 1

    return np.concatenate((l_1, l_2), axis=1)
