"""
Security Analytics - Homework 3
Neural Network from Scratch: Understanding Gradient Descent

OBJECTIVE:
----------
Build a simple neural network from scratch.
Experiment with different learning rates to understand their impact on training.

REQUIREMENTS:
-------------
1. Use random seed 42
2. Generate 1000 synthetic samples with 5 features
3. Split data: 60% train, 20% validation, 20% test
4. Neural network architecture:
   - Input: 5 features
   - Hidden layer: 8 neurons with ReLU
   - Output: 1 neuron with Sigmoid
5. Train for 1500 iterations with THREE learning rates:
   - Low: 0.1
   - Medium: 0.5
   - High: 5.0

DELIVERABLES:
-------------
1. Plot learning curves (cost vs iteration) for all 3 learning rates
2. Plot accuracy comparison for all 3 learning rates
3. Report final accuracies on test set for each learning rate

QUESTIONS TO ANSWER (IN THE PDF SUBMISSION ONLY):
--------------------
Q1: Which learning rate performs best? Why?
Q2: What happens with the high learning rate? Explain the behavior.
Q3: What is the purpose of the validation set?
Q4: Explain how backpropagation computes gradients.

INSTRUCTIONS:
-------------
Complete the functions marked with TODO. Run the code and analyze the results.
"""

import numpy as np
import matplotlib.pyplot as plt

# Set random seed
np.random.seed(42)


# ============================================================================
# STEP 1: DATA GENERATION
# ============================================================================

def generate_data(n_samples=1000, n_features=5):
    """
    Generate synthetic data.
    Class 0: Normal 
    Class 1: Anomalous 
    """
    half = n_samples // 2
    
    X_normal = np.random.randn(half, n_features) * 1.0
    y_normal = np.zeros((half, 1))
    
    X_anomalous = np.random.randn(n_samples - half, n_features) * 1.3 + 1.2
    y_anomalous = np.ones((n_samples - half, 1))
    
    X_normal[:, 0] = X_normal[:, 0] + 0.4 * X_normal[:, 1] * X_normal[:, 2] + 0.2 * X_normal[:, 3]**2
    X_anomalous[:, 0] = X_anomalous[:, 0] + 0.6 * X_anomalous[:, 1] * X_anomalous[:, 2] + 0.3 * X_anomalous[:, 3]**2
    
    X_normal[:, 4] = 0.6 * X_normal[:, 0] + 0.4 * np.random.randn(half)
    X_anomalous[:, 4] = 0.5 * X_anomalous[:, 0] + 0.5 * np.random.randn(n_samples - half)
    
    X = np.vstack([X_normal, X_anomalous])
    y = np.vstack([y_normal, y_anomalous])
    
    indices = np.random.permutation(n_samples)
    return X[indices], y[indices]


def split_data(X, y, train_ratio=0.8, val_ratio=0.1):
    """
    Split data into train, validation, and test sets.
    """
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    return X_train, y_train, X_val, y_val, X_test, y_test


# ============================================================================
# STEP 2: ACTIVATION FUNCTIONS
# ============================================================================

def sigmoid(z):
    """
    Sigmoid activation function: σ(z) = 1 / (1 + e^(-z))
    
    TODO: Implement the sigmoid function
    Hint: Use np.exp() and np.clip() to prevent overflow
    """
    # TODO: Implement sigmoid activation
    return 1 / (1 + np.exp(-z))


def relu(z):
    """
    ReLU activation function: ReLU(z) = max(0, z)
    
    TODO: Implement the ReLU function
    Hint: Use np.maximum()
    """
    # TODO: Implement ReLU activation
    return np.maximum(0,z)


def relu_derivative(z):
    """
    Derivative of ReLU function: 1 if z > 0, else 0
    
    TODO: Implement the derivative of ReLU
    Hint: Use boolean indexing and convert to float
    """
    # TODO: Implement ReLU derivative
    return (z > 0).astype(float)


# ============================================================================
# STEP 3: NEURAL NETWORK
# ============================================================================

def initialize_parameters(n_input, n_hidden):
    """
    Initialize weights and biases with small random values.
    
    TODO: Initialize the following parameters:
    - W1: weight matrix for input to hidden layer (shape: n_input x n_hidden)
    - b1: bias vector for hidden layer (shape: 1 x n_hidden)
    - W2: weight matrix for hidden to output layer (shape: n_hidden x 1)
    - b2: bias vector for output layer (shape: 1 x 1)
    
    Hint: Use np.random.randn() * 0.1 for weights and np.zeros() for biases
    """
    # TODO: Initialize W1
    W1 = np.random.randn(n_input, n_hidden) * 0.1
    
    # TODO: Initialize b1
    b1 = np.zeros((1, n_hidden))
    
    # TODO: Initialize W2
    W2 = np.random.randn(n_hidden, 1) * 0.1
    
    # TODO: Initialize b2
    b2 = np.zeros((1, 1))

    # TODO: Implement return
    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}


def forward_propagation(X, params):
    """
    Forward pass: X -> Hidden(ReLU) -> Output(Sigmoid)
    Returns predictions and cache for backprop.
    
    TODO: Implement forward propagation
    
    For hidden layer:
    1. Compute Z1 = X @ W1 + b1
    2. Compute A1 = relu(Z1)
    
    For output layer:
    3. Compute Z2 = A1 @ W2 + b2
    4. Compute A2 = sigmoid(Z2)
    
    Hint: Use np.dot() or @ for matrix multiplication
    """
    W1, b1 = params["W1"], params["b1"]
    W2, b2 = params["W2"], params["b2"]

    # TODO: Implement hidden layer forward pass
    Z1 = X @ W1 + b1
    A1 = relu(Z1)

    # TODO: Implement output layer forward pass
    Z2 = A1 @ W2 + b2
    A2 = sigmoid(Z2)
    
    # TODO: Implement cache
    cache = {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2':A2}
    return A2, cache


def compute_cost(A2, y):
    """
    Binary cross-entropy loss.
    
    TODO: Implement the binary cross-entropy cost function:
    cost = -mean(y * log(A2) + (1-y) * log(1-A2))
    
    Hint: Use np.log() and np.mean(). Add epsilon=1e-8 to avoid log(0)
    """
    m = y.shape[0]
    epsilon = 1e-8
    
    # TODO: Compute binary cross-entropy cost
    cost = -np.mean(y * np.log(A2 + epsilon) + (1 - y) * np.log(1 - A2 + epsilon))
    
    return cost


def backward_propagation(X, y, params, cache):
    """
    Backward pass: compute gradients using chain rule.
    
    TODO: Implement backpropagation
    
    For output layer:
    1. dZ2 = A2 - y
    2. dW2 = (A1.T @ dZ2) / m
    3. db2 = mean of dZ2 (keep dims)
    
    For hidden layer:
    4. dA1 = dZ2 @ W2.T
    5. dZ1 = dA1 * relu_derivative(Z1)
    6. dW1 = (X.T @ dZ1) / m
    7. db1 = mean of dZ1 (keep dims)
    
    Hint: Use np.sum() with axis=0 and keepdims=True for bias gradients
    """
    m = X.shape[0]
    
    # TODO: Compute output layer gradients
    dZ2 = None  # REPLACE THIS
    dW2 = None  # REPLACE THIS
    db2 = None  # REPLACE THIS
    
    # TODO: Compute hidden layer gradients
    dA1 = None  # REPLACE THIS
    dZ1 = None  # REPLACE THIS
    dW1 = None  # REPLACE THIS
    db1 = None  # REPLACE THIS
    
    # TODO: Implement return
    return {} # REPLACE THIS


def update_parameters(params, grads, learning_rate):
    """
    Gradient descent update: θ = θ - α * ∇θ
    
    TODO: Update each parameter using its gradient
    
    For each parameter:
    param = param - learning_rate * grad
    """
    # TODO: Update W1
    params['W1'] = None  # REPLACE THIS
    
    # TODO: Update b1
    params['b1'] = None  # REPLACE THIS
    
    # TODO: Update W2
    params['W2'] = None  # REPLACE THIS
    
    # TODO: Update b2
    params['b2'] = None  # REPLACE THIS
    
    return params


def predict(X, params):
    """Make binary predictions (0 or 1)."""
    A2, _ = forward_propagation(X, params)
    return (A2 > 0.5).astype(int)


def compute_accuracy(X, y, params):
    """Compute classification accuracy."""
    predictions = predict(X, params)
    return np.mean(predictions == y) * 100


# ============================================================================
# STEP 4: TRAINING FUNCTION
# ============================================================================

def train_network(X_train, y_train, X_val, y_val, learning_rate, n_iterations=1500, n_hidden=8):
    """
    Train neural network and return training history.
    
    TODO: Implement the training loop
    
    For each iteration:
    1. Forward propagation
    2. Compute cost
    3. Backward propagation
    4. Update parameters
    5. Track training and validation metrics
    """
    n_input = X_train.shape[1]
    params = initialize_parameters(n_input, n_hidden)
    
    train_costs = []
    val_costs = []
    train_accs = []
    val_accs = []
    
    for i in range(n_iterations):
        # TODO: Forward propagation
        A2, cache = None, None  # REPLACE THIS
        
        # TODO: Compute training cost
        train_cost = None  # REPLACE THIS
        train_costs.append(train_cost)
        
        # TODO: Backward propagation
        grads = None  # REPLACE THIS
        
        # TODO: Update parameters
        params = None  # REPLACE THIS
        
        # Validation metrics (provided)
        A2_val, _ = forward_propagation(X_val, params)
        val_cost = compute_cost(A2_val, y_val)
        val_costs.append(val_cost)
        
        # Compute accuracies every 100 iterations
        if i % 100 == 0:
            train_acc = compute_accuracy(X_train, y_train, params)
            val_acc = compute_accuracy(X_val, y_val, params)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
    
    history = {
        'train_costs': train_costs,
        'val_costs': val_costs,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'params': params
    }
    
    return history


# ============================================================================
# STEP 5: VISUALIZATION (PROVIDED - NO CHANGES NEEDED)
# ============================================================================

def plot_learning_curves(histories, learning_rates):
    """Plot cost curves for all learning rates."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, (history, lr) in enumerate(zip(histories, learning_rates)):
        ax = axes[idx]
        ax.plot(history['train_costs'], label='Train', linewidth=2)
        ax.plot(history['val_costs'], label='Validation', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Cost')
        ax.set_title(f'Learning Rate = {lr}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('learning_curves.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_accuracy_comparison(histories, learning_rates, X_test, y_test):
    """Plot accuracy comparison across learning rates."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy during training
    for history, lr in zip(histories, learning_rates):
        iterations = np.arange(0, len(history['train_accs'])) * 100
        ax1.plot(iterations, history['val_accs'], marker='o', linewidth=2, label=f'LR={lr}')
    
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Validation Accuracy (%)')
    ax1.set_title('Validation Accuracy During Training')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Final test accuracy
    test_accs = []
    for history in histories:
        test_acc = compute_accuracy(X_test, y_test, history['params'])
        test_accs.append(test_acc)
    
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    bars = ax2.bar(range(len(learning_rates)), test_accs, color=colors, alpha=0.7)
    ax2.set_xticks(range(len(learning_rates)))
    ax2.set_xticklabels([f'{lr}' for lr in learning_rates])
    ax2.set_xlabel('Learning Rate')
    ax2.set_ylabel('Test Accuracy (%)')
    ax2.set_title('Final Test Accuracy')
    ax2.set_ylim([0, 100])
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, acc in zip(bars, test_accs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# MAIN EXECUTION (PROVIDED - NO CHANGES NEEDED)
# ============================================================================

def main():
    print("=" * 70)
    print("Neural Network from Scratch: Learning Rate Comparison")
    print("=" * 70)
    print()
    
    # Generate data
    print("STEP 1: Generating synthetic data...")
    X, y = generate_data(n_samples=1000, n_features=5)
    print(f"  Total samples: {len(X)}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Normal traffic: {np.sum(y == 0)} | Anomalous traffic: {np.sum(y == 1)}")
    print()
    
    # Split data
    print("STEP 2: Splitting data...")
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)
    print(f"  Train: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")
    print()
    
    # Train with different learning rates
    print("STEP 3: Training networks with different learning rates...")
    print("-" * 70)
    
    learning_rates = [0.01, 0.5, 5.0]
    histories = []
    
    for lr in learning_rates:
        print(f"\nTraining with learning rate = {lr}...")
        history = train_network(X_train, y_train, X_val, y_val, 
                               learning_rate=lr, n_iterations=1500)
        histories.append(history)
        
        # Print final metrics
        train_acc = compute_accuracy(X_train, y_train, history['params'])
        val_acc = compute_accuracy(X_val, y_val, history['params'])
        test_acc = compute_accuracy(X_test, y_test, history['params'])
        
        print(f"  Final train cost: {history['train_costs'][-1]:.4f}")
        print(f"  Final val cost: {history['val_costs'][-1]:.4f}")
        print(f"  Test accuracy: {test_acc:.2f}%")
    
    print("-" * 70)
    print()
    
    # Visualize results
    print("STEP 4: Generating visualizations...")
    plot_learning_curves(histories, learning_rates)
    plot_accuracy_comparison(histories, learning_rates, X_test, y_test)
    print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nTest Accuracy by Learning Rate:")
    for lr, history in zip(learning_rates, histories):
        test_acc = compute_accuracy(X_test, y_test, history['params'])
        print(f"  LR = {lr:5.2f} → Test Accuracy = {test_acc:5.2f}%")


if __name__ == "__main__":
    main()

