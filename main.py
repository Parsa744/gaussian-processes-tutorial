# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, ConstantKernel as C
from ipywidgets import interact, FloatSlider, IntSlider, Dropdown

# Set a style for the plots
plt.style.use('seaborn-v0_8-whitegrid')


# 1. Define the true function
def true_function(x):
    """The underlying function we want to model (a sine wave)."""
    return np.sin(2.5 * x)


# 2. Create the main plotting function controlled by the widgets
def plot_regression_comparison(
        n_samples=25,
        noise_level=0.25,
        kernel_type='RBF',
        length_scale=1.0
):
    """
    Generates data, fits three different regression models, and plots the results.
    """

    # --- Data Generation ---
    X_train = np.linspace(-3, 3, n_samples).reshape(-1, 1)
    y_train = true_function(X_train).ravel() + np.random.normal(0, noise_level, n_samples)
    X_plot = np.linspace(-4, 4, 200).reshape(-1, 1)

    # --- Model Fitting ---
    # a) Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_lr_pred = lr_model.predict(X_plot)

    # b) Support Vector Regression (a non-linear model)
    svr_model = SVR(kernel='rbf', C=1.0, gamma='auto')
    svr_model.fit(X_train, y_train)
    y_svr_pred = svr_model.predict(X_plot)

    # c) Gaussian Process Regression
    # Select the kernel based on user input
    if kernel_type == 'RBF':
        kernel = C(1.0) * RBF(length_scale=length_scale, length_scale_bounds=(1e-2, 1e3))
    elif kernel_type == 'Matern':
        # nu=1.5 is a common choice, resulting in a once-differentiable function
        kernel = C(1.0) * Matern(length_scale=length_scale, length_scale_bounds=(1e-2, 1e3), nu=1.5)
    elif kernel_type == 'RationalQuadratic':
        kernel = C(1.0) * RationalQuadratic(length_scale=length_scale, alpha=1.0)

    gpr_model = GaussianProcessRegressor(kernel=kernel, alpha=noise_level ** 2, n_restarts_optimizer=10)
    gpr_model.fit(X_train, y_train)
    y_gpr_pred, sigma = gpr_model.predict(X_plot, return_std=True)

    # --- Visualization ---
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 7), sharey=True)
    fig.suptitle('Comparing Regression Models', fontsize=20)

    # Plot for Linear Regression
    ax1.set_title('Linear Regression')
    ax1.plot(X_plot, true_function(X_plot), 'r:', label='True Function')
    ax1.scatter(X_train, y_train, c='b', s=50, zorder=10, edgecolors='k', label='Noisy Data')
    ax1.plot(X_plot, y_lr_pred, 'g-', lw=2, label='LR Prediction')
    ax1.set_xlabel('Input (x)');
    ax1.set_ylabel('Output (y)')
    ax1.legend();
    ax1.set_xlim(-4, 4);
    ax1.set_ylim(-3, 3)
    ax1.text(-3.8, 2.4, "❌ Fails to capture non-linearity.", fontsize=11)

    # Plot for Support Vector Regression
    ax2.set_title('Support Vector Regression (SVR)')
    ax2.plot(X_plot, true_function(X_plot), 'r:', label='True Function')
    ax2.scatter(X_train, y_train, c='b', s=50, zorder=10, edgecolors='k', label='Noisy Data')
    ax2.plot(X_plot, y_svr_pred, 'm-', lw=2, label='SVR Prediction')
    ax2.set_xlabel('Input (x)');
    ax2.legend()
    ax2.set_xlim(-4, 4);
    ax2.set_ylim(-3, 3)
    ax2.text(-3.8, 2.4, "✅ Captures non-linearity.\n❌ No uncertainty estimate.", fontsize=11)

    # Plot for Gaussian Process Regression
    ax3.set_title(f'Gaussian Process (Kernel: {kernel_type})')
    ax3.plot(X_plot, true_function(X_plot), 'r:', label='True Function')
    ax3.scatter(X_train, y_train, c='b', s=50, zorder=10, edgecolors='k', label='Noisy Data')
    ax3.plot(X_plot, y_gpr_pred, 'c-', lw=2, label='GPR Mean Prediction')
    ax3.fill_between(X_plot.ravel(), y_gpr_pred - 1.96 * sigma, y_gpr_pred + 1.96 * sigma,
                     alpha=0.3, color='c', label='95% Confidence Interval')
    ax3.set_xlabel('Input (x)');
    ax3.legend()
    ax3.set_xlim(-4, 4);
    ax3.set_ylim(-3, 3)
    ax3.text(-3.8, 2.4, "✅ Captures non-linearity.\n✅ Provides uncertainty!", fontsize=11)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# 3. Create the interactive widgets
interact(
    plot_regression_comparison,
    n_samples=IntSlider(min=5, max=150, step=5, value=25, description='Data Points:'),
    noise_level=FloatSlider(min=0.0, max=1.0, step=0.05, value=0.25, description='Noise Level:'),
    kernel_type=Dropdown(options=['RBF', 'Matern', 'RationalQuadratic'], value='RBF', description='GPR Kernel:'),
    length_scale=FloatSlider(min=0.1, max=5.0, step=0.1, value=1.0, description='GPR Length Scale:')
);