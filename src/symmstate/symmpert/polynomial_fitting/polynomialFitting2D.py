#!/usr/bin/env python3
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def parse_args():
    if len(sys.argv) < 3:
        print("Usage: python script.py <data_file> <powers>")
        print("Example: python script.py data.txt 2 4")
        sys.exit(1)
    return sys.argv[1], [int(p) for p in sys.argv[2:]]


def load_data(filename):
    data = np.loadtxt(filename)
    return data[:, 0], data[:, 1]


def create_polynomial_function(powers):
    def polynomial(x, *params):
        return (
            sum(param * x**power for param, power in zip(params, powers)) + params[-1]
        )

    return polynomial


def fit_and_evaluate(x, y, func, p0):
    popt, _ = curve_fit(func, x, y, p0=p0, maxfev=10000)
    y_pred = func(x, *popt)
    mse = np.mean((y - y_pred) ** 2)
    r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
    return popt, mse, r2, y_pred


def plot_results(x, y, func, popt, powers, step, filename):
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color="blue", label="Data points")

    x_smooth = np.linspace(min(x), max(x), 1000)
    y_smooth = func(x_smooth, *popt)

    plt.plot(x_smooth, y_smooth, color="red", label="Fitted polynomial")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Polynomial Fit (Step {step})")
    plt.legend()

    y_range = max(y) - min(y)
    plt.ylim(min(y) - 0.1 * y_range, max(y) + 0.1 * y_range)

    plt.savefig(filename)
    plt.close()


def print_equation(popt, powers):
    equation = "y = "
    for coef, power in zip(popt[:-1], powers):
        equation += f"{coef:.6e}x^{power} + "
    equation += f"{popt[-1]:.6e}"
    return equation


def main():
    data_file, powers = parse_args()
    x, y = load_data(data_file)

    for step, power in enumerate(powers, 1):
        current_powers = powers[:step]
        poly_func = create_polynomial_function(current_powers)

        if step == 1:
            initial_guess = [1.0] * (step + 1)  # +1 for the constant term
        else:
            initial_guess = list(previous_popt) + [0.0]  # Add a new parameter

        popt, mse, r2, y_pred = fit_and_evaluate(x, y, poly_func, initial_guess)

        print(f"Step {step}: Fitting terms up to x^{power}")
        print(print_equation(popt, current_powers))
        print(f"MSE: {mse:.6e}")
        print(f"R-squared: {r2:.6f}\n")

        plot_filename = f"polynomial_fit_step_{step}.png"
        plot_results(x, y, poly_func, popt, current_powers, step, plot_filename)
        print(f"Plot saved as {plot_filename}")

        previous_popt = popt

    print("\nFitting process completed.")


if __name__ == "__main__":
    main()
