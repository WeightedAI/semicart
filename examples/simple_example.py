#!/usr/bin/env python
"""
Simple example demonstrating the usage of SemiCART.

This example compares SemiCART with standard CART on the Iris dataset
and shows how SemiCART can improve predictive performance.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from semicart import SemiCART
from semicart.metrics.evaluation import evaluate_model


def main():
    """Run a simple SemiCART example on the Iris dataset."""
    print("Loading Iris dataset...")
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split the data
    test_size = 0.3
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=42
    )
    
    print(f"Data split: {len(X_train)} training samples, {len(X_test)} test samples")
    
    # Create and fit standard CART
    print("\nTraining standard CART...")
    cart = DecisionTreeClassifier(random_state=42)
    cart.fit(X_train, y_train)
    
    # Predict with standard CART
    y_pred_cart = cart.predict(X_test)
    y_proba_cart = cart.predict_proba(X_test)
    
    # Evaluate standard CART
    cart_metrics = evaluate_model(y_test, y_pred_cart, y_proba_cart)
    print(f"Standard CART accuracy: {cart_metrics['accuracy']:.4f}")
    
    # Try different k values for SemiCART
    k_values = [1, 3, 5, 7]
    semi_accuracies = []
    
    for k in k_values:
        print(f"\nTraining SemiCART with k={k}...")
        
        # Create and fit SemiCART
        semicart = SemiCART(k_neighbors=k, random_state=42)
        semicart.fit(X_train, y_train, X_test)
        
        # Predict with SemiCART
        y_pred_semi = semicart.predict(X_test)
        y_proba_semi = semicart.predict_proba(X_test)
        
        # Evaluate SemiCART
        semi_metrics = evaluate_model(y_test, y_pred_semi, y_proba_semi)
        semi_accuracies.append(semi_metrics['accuracy'])
        
        print(f"SemiCART (k={k}) accuracy: {semi_metrics['accuracy']:.4f}")
        print(f"Improvement over CART: {semi_metrics['accuracy'] - cart_metrics['accuracy']:.4f}")
    
    # Find the best k value
    best_k_idx = np.argmax(semi_accuracies)
    best_k = k_values[best_k_idx]
    best_accuracy = semi_accuracies[best_k_idx]
    
    print(f"\nBest SemiCART configuration: k={best_k} with accuracy {best_accuracy:.4f}")
    print(f"Improvement over standard CART: {best_accuracy - cart_metrics['accuracy']:.4f}")
    
    # Plot the results
    print("\nGenerating plot...")
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, semi_accuracies, marker='o', label='SemiCART')
    plt.axhline(y=cart_metrics['accuracy'], linestyle='--', color='r', label='Standard CART')
    plt.xlabel('k neighbors')
    plt.ylabel('Accuracy')
    plt.title('SemiCART vs Standard CART on Iris Dataset')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('semicart_example.png')
    print("Plot saved as 'semicart_example.png'")


if __name__ == "__main__":
    main() 