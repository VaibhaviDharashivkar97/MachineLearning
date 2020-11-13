"""
Author@VaibhaviDharashivkar
"""
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle


class LogisticRegression:
    def __init__(self):
        self.weights = np.zeros((2, 1), dtype=float)

    def training(self, x_train, y_train, itering):
        for i in range(itering):
            op = self.predicting(x_train)
            err = y_train - op
            updates = np.dot(x_train.T, err * self.sigmoid(op, True))
            self.weights = self.weights + updates

    def predicting(self, ip):
        return self.sigmoid(np.dot(ip, self.weights), False)

    def sigmoid(self, x, der):
        if der:
            return x * (1 - x)
        else:
            return 1 / (1 + np.exp(-x))


def main(filename):
    MFCCs_10 = []
    MFCCs_17 = []
    Species = []
    data = np.genfromtxt(filename, dtype=None, delimiter=',', skip_header=1, unpack=True)
    for d in data:
        for var in range(3):
            if var == 0:
                MFCCs_10.append(d[var])
            elif var == 1:
                MFCCs_17.append(d[var])
            else:
                if d[var] == b'HylaMinuta':
                    Species.append(0)
                else:
                    Species.append(1)
    model = LogisticRegression()
    tempx_test = []
    tempx_train = []
    tempy_test = []
    tempy_train = []
    n = (1 / 5) * len(Species)
    r = random.sample(range(len(Species)), len(Species))
    count = 0
    for i in r:
        count += 1
        if count >= n:
            tempx_train.append([MFCCs_10[i], MFCCs_17[i]])
            tempy_train.append([Species[i]])
        else:
            tempx_test.append([MFCCs_10[i], MFCCs_17[i]])
            tempy_test.append([Species[i]])

    x_train = np.asarray(tempx_train)
    x_test = np.asarray(tempx_test)
    y_train = np.asarray(tempy_train)
    y_test = np.asarray(tempy_test)

    model.training(x_train, y_train, 15000)
    predicted_op = []
    for t in x_test:
        y_predicted = model.predicting(t)
        if abs(y_predicted - 0) > (1 - y_predicted):
            predicted_op.append(1)
        else:
            predicted_op.append(0)
    accepted = 0
    for i in range(len(predicted_op)):
        if [predicted_op[i]] == y_test[i]:
            accepted = accepted + 1
    print(filename.split('.')[0]+":")
    print(f"Accuracy = {accepted / len(y_test) * 100}")
    theta1 = [1]
    for i in model.weights:
        theta1.append(i[0])
    pickle.dump(theta1, open(filename.split('.')[0]+"Model", "wb"))
    fle = open("model", 'rb')
    theta = pickle.load(fle)
    print("theta:", theta)
    count = 0
    for i in Species:
        if i == 0:
            count += 1

    x = x_train[:, 1]
    y = -((theta[0] / theta[2]) / (theta[0] / theta[1])) * x + (-theta[0] / theta[2])

    colors = []
    for i in Species:
        if i == 0:
            colors.append('red')
        else:
            colors.append('blue')
    fig = plt.figure()
    plt.title(filename.split('.')[0])
    plt.xlim(-0.5, 0.5)
    plt.ylim(-0.5, 0.5)
    ax = fig.add_subplot(111)

    plt.xlabel('MFCCs_10')
    plt.ylabel('MFCCs_17')
    ax.scatter(MFCCs_10, MFCCs_17, s=100, alpha=0.7, color=colors, edgecolors='black')
    red_patch = mpatches.Patch(color='red', label='HylaMinuta')
    blue_patch = mpatches.Patch(color='blue', label='HypsiboasCinerascens')
    fig1, = ax.plot(x, y, color='green', label='Decision Boundry')
    plt.legend(handles=[fig1, red_patch, blue_patch], loc=0)
    plt.savefig(filename.split('.')[0] + "_log_reg.png")
    plt.show()


if __name__ == '__main__':
    main('Frogs-subsample.csv')
    main('Frogs.csv')
