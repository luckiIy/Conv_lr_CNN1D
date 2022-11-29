from libsvm.svmutil import *
from libsvm.svm import *
import numpy as np


from models.utils import get_data

def svm_baseline():
    traindata, testdata = get_data('usps')
    y, x = traindata
    yt, xt = testdata
    # acc = 0
    # if is_trainsvm:
    #     model = svm_train(y, x, '-t 1')
    #     p_label, p_acc, p_val = svm_predict(yt[:], xt[:], model)
    #     acc = p_acc[0]
    model = svm_train(y, x, '-t 1')
    p_label, p_acc, p_val = svm_predict(yt[:], xt[:], model)
    return p_acc, traindata, testdata



if __name__ == '__main__':
    acc, _, _ = svm_baseline()
    print(acc[0])
# y, x = svm_read_problem('dataset/gisette/gisette_scale', return_scipy=True)
# yt, xt = svm_read_problem('dataset/gisette/gisette_scale.t', return_scipy=True)
# model = svm_train(y, x,'-t 1')
# p_label, p_acc, p_val = svm_predict(yt[:], xt[:], model)
# print(p_acc)


# y, x = [1, -1], [{1: 1, 2: 1}, {1: -1, 2: -1}]
# prob = svm_problem(y, x)
# param = svm_parameter('-t 0 -c 4 -b 1')
# model = svm_train(prob, param)
# yt = [1]
# xt = [{1: 1, 2: 1}]
# p_label, p_acc, p_val = svm_predict(yt, xt, model)
# print(p_label)

