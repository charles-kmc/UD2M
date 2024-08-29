import numpy as np
import os
import csv 

def write_mat_to_file(mat, file, dir = './' ,sep = ',', headers = None, mode = "w"):
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok = True)
    file = os.path.join(dir, file)
    try:
        mat = np.array(mat, ndmin = 2)
    except:
        raise ValueError("Matrix should be broadcastable to a numpy.ndarray")
    M, N = mat.shape
    file = open(file, mode)
    if headers is not None:
        for n in range(N):
            file.write(str(headers[n]))
            if n < N-1:
                file.write(sep)
        file.write('\n')
    for m in range(M):
        for n in range(N):
            file.write(str(mat[m, n]))
            if n < N-1:
                file.write(sep)
        file.write('\n')

def write_dict_to_file(D, path):
    file = open(path, "w")
    w = csv.writer(file)
    w.writerow(D.keys())
    w.writerows(zip(*D.values()))
    file.close()

def write_config_to_file(config, path):
    with open(path, "w") as file:
        for k, v in config.__dict__.items():
            try:
               file.write(k + str(v) + '\n')
            except:
                print(f'WARNING: unable to store config item {k}')



def load_file_from_dict(path):
    with open(path) as file:
        reader = csv.DictReader(file)
        for i, row in enumerate(reader):
            if i == 0:
                out = {k:[] for k in row.keys()}
            for k, v in row.items():
                out[k].append(v)
    return out
