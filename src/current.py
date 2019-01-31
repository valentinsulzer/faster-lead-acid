import numpy as np
import matplotlib.pyplot as plt

def current(t, Ibar):
    y = Ibar
    # result = []
    # for _ in t:
    #     result.append(y)
    #     y += np.random.normal(scale=Ibar/30)
    # return np.array(result)
    return Ibar + np.random.normal(scale=Ibar/5, size=t.shape)

if __name__ == "__main__":
    Ibar = 1
    t = np.linspace(0,1,100)
    plt.plot(t, current(t, Ibar))
    plt.ylim([0,2])
    plt.show()
