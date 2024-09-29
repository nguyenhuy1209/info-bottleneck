from itertools import product

import matplotlib.pyplot as plt
import numpy as np


# Generate dataset
def generate_dataset():
    input_dim = 12
    X = np.array(list(product([0, 1], repeat=input_dim)))
    f_X = np.sum(X, axis=1)
    theta = np.mean(f_X)
    gamma = 5.0  # Steeper sigmoid for clearer separation
    P_Y_given_X = 1 / (1 + np.exp(-gamma * (f_X - theta)))
    Y = (np.random.rand(len(X)) < P_Y_given_X).astype(int)
    Y = Y.reshape(-1, 1)
    return X, Y


# Calculate probabilities
def calculate_probabilities(X, Y):
    N, input_dim = X.shape
    unique_X, counts_X = np.unique(X, axis=0, return_counts=True)
    P_X = counts_X / N
    P_Y_given_X = np.zeros((len(unique_X), 2))  # Two columns for P(Y=0|X) and P(Y=1|X)
    for i, x in enumerate(unique_X):
        mask = np.all(X == x, axis=1)
        P_Y_given_X[i, 0] = np.mean(Y[mask] == 0)  # P(Y=0|X)
        P_Y_given_X[i, 1] = np.mean(Y[mask] == 1)  # P(Y=1|X)
    return P_X, P_Y_given_X


# Information Bottleneck algorithm
def information_bottleneck(PX, PY_given_X, beta, max_iter=10000, tol=1e-6):
    N, M = PY_given_X.shape
    K = M  # Use a smaller number of clusters initially
    P_T_given_X = np.random.rand(N, K)
    P_T_given_X /= P_T_given_X.sum(axis=1, keepdims=True)
    is_proper_prob = np.all(np.isclose(np.sum(P_T_given_X, axis=1), 1.0))
    print(is_proper_prob)

    for iteration in range(max_iter):
        print(f"Iteration: {iteration}")

        # Calculate P_T
        P_T = (P_T_given_X * PX[:, np.newaxis]).sum(axis=0)
        is_proper_prob = np.isclose(np.sum(P_T), 1.0)
        print(f"Check proper distribution P_T: {is_proper_prob}")
        if not is_proper_prob:
            print(f"P_T is not properly calculated")
        print(P_T)

        # Calculate P_Y_given_T
        P_Y_given_T = (
            np.dot(P_T_given_X.T, PY_given_X * PX[:, np.newaxis]) / P_T[:, np.newaxis]
        )
        is_proper_prob = np.all(np.isclose(np.sum(P_Y_given_T, axis=1), 1.0))
        print(f"Check proper distribution P_Y_given_T: {is_proper_prob}")
        if not is_proper_prob:
            print(f"P_Y_given_T is not properly calculated")
        print(P_Y_given_T)

        # Calculate P_T_given_X
        log_P_T_given_X = -beta * np.sum(
            PY_given_X[:, :, np.newaxis]
            * np.log(
                np.maximum(
                    PY_given_X[:, :, np.newaxis] / np.maximum(P_Y_given_T.T, 1e-300),
                    1e-300,
                )
            ),
            axis=1,
        )
        log_P_T_given_X += np.log(P_T)
        log_P_T_given_X -= np.max(
            log_P_T_given_X, axis=1, keepdims=True
        )  # for numerical stability
        P_T_given_X_new = np.exp(log_P_T_given_X)
        P_T_given_X_new /= np.sum(P_T_given_X_new, axis=1, keepdims=True)
        is_proper_prob = np.all(np.isclose(np.sum(P_T_given_X_new, axis=1), 1.0))
        print(f"Check proper distribution P_T_given_X: {is_proper_prob}")
        if not is_proper_prob:
            print(f"P_T_given_X is not properly calculated")
        print(P_T_given_X_new)

        KL_divergence = np.sum(
            P_T_given_X_new * np.log(P_T_given_X_new / np.maximum(P_T_given_X, 1e-10))
        )
        if np.isnan(KL_divergence):
            print(f"KL divergence is lesser than 0: {KL_divergence}")
            raise
        if KL_divergence < tol:
            break

        P_T_given_X = P_T_given_X_new

    return P_T_given_X, P_T, P_Y_given_T


# Calculate mutual information
def mutual_information(P, Q):
    return np.sum(P * np.log(np.maximum(P / np.maximum(Q, 1e-10), 1e-10)))


# Calculate IB curve
def calculate_IB_curve(P_X, P_T_given_X, P_T, P_Y_given_T):
    I_X_T = np.sum(
        P_X[:, np.newaxis]
        * P_T_given_X
        * np.log(np.maximum(P_T_given_X / np.maximum(P_T, 1e-10), 1e-10))
    )
    P_Y_and_T = P_Y_given_T.T * P_T
    P_Y = np.sum(P_Y_and_T, axis=1)
    I_T_Y = np.sum(
        P_Y_and_T
        * np.log(
            np.maximum(P_Y_and_T / np.maximum(P_Y[:, np.newaxis] * P_T, 1e-10), 1e-10)
        )
    )
    return I_X_T, I_T_Y


# Generate dataset
X, Y = generate_dataset()

# Calculate probabilities
P_X, P_Y_given_X = calculate_probabilities(X, Y)

# List of beta values
beta_values = np.linspace(35, 250, 5)  # Use a wider range of beta values
beta_values = [int(values) for values in beta_values]
I_X_T_values = []
I_T_Y_values = []

for beta in beta_values:
    P_T_given_X, P_T, P_Y_given_T = information_bottleneck(P_X, P_Y_given_X, beta)
    I_X_T, I_T_Y = calculate_IB_curve(P_X, P_T_given_X, P_T, P_Y_given_T)
    I_X_T_values.append(I_X_T)
    I_T_Y_values.append(I_T_Y)

# Plot the IB curve
plt.figure(figsize=(10, 6))
for i, beta in enumerate(beta_values):
    plt.plot(I_X_T_values[i], I_T_Y_values[i], marker="o", label=f"β={beta:.1f}")
plt.plot(I_X_T_values, I_T_Y_values, "-")

plt.xlabel("I(X; T)")
plt.ylabel("I(T; Y)")
plt.title("Information Bottleneck Curve")
plt.legend()
plt.savefig("information_bottleneck_curve.png")
plt.show()
