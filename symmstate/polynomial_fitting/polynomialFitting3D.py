#!/usr/bin/env python3
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D
from itertools import combinations


def parse_args():
    if len(sys.argv) < 3:
        print("Usage: python script.py <data_file> <terms>")
        print("Example: python script.py data.txt x^2 xy y^4")
        sys.exit(1)
    return sys.argv[1], sys.argv[2:]


def load_data(filename):
    data = np.loadtxt(filename)
    return data[:, 0], data[:, 1], data[:, 2]


def scale_data(x, y, z):
    x_scaled = (x - np.min(x)) / (np.max(x) - np.min(x))
    y_scaled = (y - np.min(y)) / (np.max(y) - np.min(y))
    z_scaled = (z - np.min(z)) / (np.max(z) - np.min(z))
    return x_scaled, y_scaled, z_scaled


def create_polynomial_function(terms):
    def polynomial(xy, *params):
        x, y = xy
        result = 0
        for param, term in zip(params, terms):
            if "x" in term and "y" in term:
                x_power = term.count("x")
                y_power = term.count("y")
                result += param * (x**x_power) * (y**y_power)
            elif "x" in term:
                power = term.count("x")
                result += param * x**power
            elif "y" in term:
                power = term.count("y")
                result += param * y**power
            else:
                result += param  # constant term
        return result

    return polynomial


def fit_and_evaluate(x, y, z, func, p0):
    popt, _ = curve_fit(func, (x, y), z, p0=p0, maxfev=10000)
    z_pred = func((x, y), *popt)
    mse = np.mean((z - z_pred) ** 2)
    r2 = 1 - np.sum((z - z_pred) ** 2) / np.sum((z - np.mean(z)) ** 2)
    return popt, mse, r2, z_pred


def plot_results(x, y, z, func, popt, terms, step, filename):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(x, y, z, color="blue", label="Data points")

    x_smooth = np.linspace(0, 1, 100)
    y_smooth = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x_smooth, y_smooth)
    Z = func((X, Y), *popt)

    ax.plot_surface(X, Y, Z, color="red", alpha=0.5, label="Fitted surface")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Polynomial Surface Fit (Step {step})")

    plt.savefig(filename)
    plt.close()


def print_equation(popt, terms):
    equation = "z = "
    for coef, term in zip(popt, terms):
        if term:
            equation += f"{coef:.6e}{term} + "
        else:
            equation += f"{coef:.6e}"
    return equation.rstrip(" + ")


def stepwise_fitting(x, y, z, terms):
    best_terms = []
    best_mse = float("inf")
    best_r2 = -float("inf")
    best_popt = None

    for i in range(1, len(terms) + 1):
        for combination in combinations(terms, i):
            current_terms = list(combination)
            poly_func = create_polynomial_function(current_terms)
            initial_guess = [1.0] * len(current_terms)

            popt, mse, r2, _ = fit_and_evaluate(x, y, z, poly_func, initial_guess)

            if r2 > best_r2:
                best_terms = current_terms
                best_mse = mse
                best_r2 = r2
                best_popt = popt

        print(f"Step {i}: Best terms {', '.join(best_terms)}")
        print(print_equation(best_popt, best_terms))
        print(f"MSE: {best_mse:.6e}")
        print(f"R-squared: {best_r2:.6f}\n")

        plot_filename = f"polynomial_surface_fit_step_{i}.png"
        plot_results(
            x,
            y,
            z,
            create_polynomial_function(best_terms),
            best_popt,
            best_terms,
            i,
            plot_filename,
        )
        print(f"Plot saved as {plot_filename}")

    return best_terms, best_popt, best_mse, best_r2


def main():
    data_file, terms = parse_args()
    x, y, z = load_data(data_file)
    x_scaled, y_scaled, z_scaled = scale_data(x, y, z)

    best_terms, best_popt, best_mse, best_r2 = stepwise_fitting(
        x_scaled, y_scaled, z_scaled, terms
    )

    print("\nFinal best fit:")
    print(f"Terms: {', '.join(best_terms)}")
    print(print_equation(best_popt, best_terms))
    print(f"MSE: {best_mse:.6e}")
    print(f"R-squared: {best_r2:.6f}")


if __name__ == "__main__":
    main()
