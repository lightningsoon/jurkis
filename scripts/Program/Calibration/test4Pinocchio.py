from EyeArm_Learning import Pinocchio
import numpy as np


myP = Pinocchio()

def main():
    myP.train()
    myP.readData()
    shuffle = np.arange(0, 600, 50)
    x, y = myP.X[shuffle, :], myP.Y[shuffle, :]
    for i in range(len(x)):
        print('x:', x[i], 'y:', y[i], 'pred', myP.predict(x[i]))
    pass

if __name__ == '__main__':
    main()
    print [278, 354, 595],myP.predict([278, 354, 595])
