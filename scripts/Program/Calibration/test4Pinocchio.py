from EyeArm_Learning import Pinocchio
import numpy as np
def main():
    myP = Pinocchio()
    myP.train()
    myP.readData()
    shuffle = np.arange(0, 100, 10)
    x, y = myP.X[shuffle, :], myP.Y[shuffle, :]
    for i in range(len(x)):
        print('x:', x[i], 'y:', y[i], 'pred', myP.predict(x[i]))
    pass
if __name__ == '__main__':
    main()