import numpy as np

ENDPOINT_A = 0.01
ENDPOINT_B = np.pi/2 - 0.01
TOL = 10**-3
MAX_ITER = 100


def f(x):
    return 1.0/(np.sin(x)) - np.tan(x)

def main():

    a = ENDPOINT_A
    b = ENDPOINT_B
    f_a = f(a)

    print("i\t   a_i\t\t   b_i\t   x_i = (a_i + b_i)/2\t   f(a_i)*f(x_i) > 0  \t error = |a_i - b_i|/2")
    print("-------------------------------------------------------------------------------------------------------")
    i = 0
    while i <= MAX_ITER:
        previous_p = p_i if i > 0 else np.inf
        p_i = (a + b)/2.0
        f_p = f(p_i)

        print("%d\t%f\t%f\t%f\t\t%s\t\t\t%f" % (i, a, b, p_i, "True" if f_a * f_p > 0 else "False", abs(a - b)/2))

        # relative error cannot be smaller than TOL
        if f_p == 0 or abs(previous_p - p_i)/abs(p_i) < TOL:
            print("Approximate root: {}".format(str(p_i)))
            break

        if f_a * f_p > 0:
            a = p_i
            f_a = f_p
        else:
            b = p_i

        i = i + 1

if __name__ == "__main__":
    main()
