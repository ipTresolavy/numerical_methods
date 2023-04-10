import numpy as np

ENDPOINT_A = 0
ENDPOINT_B = 0.5
TOL = 10**-2
MAX_ITER = 100


def f(x):
    return 5*np.pi -10*np.arcsin(x) -10*x*np.sqrt(1-x**2) -12.4

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

        print("%d\t%f\t%f\t%f\t\t%s\t\t\t%f" % (i+1, a, b, p_i, "True" if f_a * f_p > 0 else "False", abs(a - b)/2))

        # relative error cannot be smaller than TOL
        if f_p == 0 or abs(a-b)/2 < TOL:
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
