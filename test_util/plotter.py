import numpy as np
from scipy import interpolate
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def plot_image_vary_data():
    x = [20, 40, 50, 60, 70, 100, 200, 300, 500]
    y = [0.563729, 0.370127, 0.406534, 0.288515, 0.301678, 0.219797, 0.187413, 0.153894, 0.139735]

    y = [x*100 for x in y]
    scale_y = 100

    fig = plt.figure()

    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/scale_y))
    ax1.yaxis.set_major_formatter(ticks_y)
    ax2.yaxis.set_major_formatter(ticks_y)

    opt, pcov = curve_fit(model_func, x, y, (1., 1.e-5, 1.))
    a, k, b = opt

    x2 = np.linspace(0, 550)
    y2 = model_func(x2, a, k, b)

    ax1.scatter(x, y)
    # ax1.plot(x2, y2, 'b')

    x = [1, 2, 5, 10, 20]
    y = [0.25703, 0.09052, 0.03159, 0.01525, 0.009677]

    y = [x*100 for x in y]

    opt, pcov = curve_fit(model_func, x, y, (1., 1.e-5, 1.))
    a, k, b = opt

    x2 = np.linspace(0, 25)
    y2 = model_func(x2, a, k, b)

    ax2.scatter(x, y)
    # ax2.plot(x2, y2, 'r')

    ax1.set_ylabel("Average Runtime [s/Image/Process]")
    ax1.set_xlabel("Number of Images [#]")
    ax2.set_xlabel("Number of Processes [#]")

    plt.show()

def plot_process_diff():
    x = [1, 2, 5, 10, 20]
    y = [0.25703, 0.09052, 0.03159, 0.01525, 0.009677]

    y = [x*100 for x in y]

    opt, pcov = curve_fit(model_func, x, y, (2.5, 1.e-10, 1.))
    a, k, b = opt

    x2 = np.linspace(0, 25)
    y2 = model_func(x2, a, k, b)

    plt.scatter(x, y)
    plt.plot(x2, y2, 'r')
    plt.show()


def plot_external(y):
    x = [64, 32, 16, 8]
    f = interpolate.interp1d(x, y)

    xnew = np.linspace(8, 64)

    plt.plot(xnew, f(xnew))
    plt.scatter(x, y)
    plt.show()


def model_func(x, a, k, b): #copy pasta straight from https://stackoverflow.com/questions/37713691/python-fitting-exponential-decay-curve-from-recorded-values
    return a*np.exp(-k*x) + b


if __name__ == '__main__':
    ch = [0.169456, 0.193516, 0.200455, 0.246939]
    us = [0.09552, 0.107803, 0.134684, 0.133583]
    # plot_external(ch)
    # plot_external(us)
    plot_image_vary_data()