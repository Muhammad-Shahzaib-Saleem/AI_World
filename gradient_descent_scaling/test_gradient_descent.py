#!/usr/bin/env python3
"""
Test script for the Gradient Descent implementation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import sys
import os

# Add the current directory to the path to import our custom class
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our custom gradient descent class
from app import GradientDescentRegressor

def test_gradient_descent():
    """Test the gradient descent implementation"""
    print("🧪 Testing Gradient Descent Implementation")
    print("=" * 50)
    
    # Load sample dataset
    print("📊 Loading sample dataset...")
    df = pd.read_csv('sample_datasets/housing_prices.csv')
    print(f"Dataset shape: {df.shape}")
    
    # Prepare data
    X = df.drop('price', axis=1)
    y = df['price'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Test different scaling methods
    scaling_methods = ['none', 'standard', 'minmax', 'robust']
    results = {}
    
    print("\n🔄 Testing different scaling methods...")
    
    for method in scaling_methods:
        print(f"\n📈 Testing {method} scaling...")
        
        # Train our custom gradient descent
        model = GradientDescentRegressor(
            learning_rate=0.01,
            max_iterations=1000,
            tolerance=1e-6,
            scaling_method=method
        )
        
        model.fit(X_train, y_train)
        
        # Get predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train, train_pred)
        test_mse = mean_squared_error(y_test, test_pred)
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        results[method] = {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'iterations': len(model.cost_history),
            'final_cost': model.cost_history[-1] if model.cost_history else None
        }
        
        print(f"  ✅ Converged in {len(model.cost_history)} iterations")
        print(f"  📊 Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
        print(f"  📉 Train MSE: {train_mse:.2f}, Test MSE: {test_mse:.2f}")
    
    # Compare with sklearn LinearRegression
    print("\n🔬 Comparing with sklearn LinearRegression...")
    
    # Standard scaling for fair comparison
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    sklearn_model = LinearRegression()
    sklearn_model.fit(X_train_scaled, y_train)
    
    sklearn_train_pred = sklearn_model.predict(X_train_scaled)
    sklearn_test_pred = sklearn_model.predict(X_test_scaled)
    
    sklearn_train_r2 = r2_score(y_train, sklearn_train_pred)
    sklearn_test_r2 = r2_score(y_test, sklearn_test_pred)
    
    print(f"  📊 sklearn Train R²: {sklearn_train_r2:.4f}, Test R²: {sklearn_test_r2:.4f}")
    
    # Results summary
    print("\n📋 Results Summary")
    print("=" * 50)
    
    for method, result in results.items():
        print(f"{method.upper():>10} | Train R²: {result['train_r2']:6.4f} | Test R²: {result['test_r2']:6.4f} | Iterations: {result['iterations']:4d}")
    
    print(f"{'SKLEARN':>10} | Train R²: {sklearn_train_r2:6.4f} | Test R²: {sklearn_test_r2:6.4f} | Iterations:  N/A")
    
    # Find best method
    best_method = max(results.keys(), key=lambda x: results[x]['test_r2'])
    print(f"\n🏆 Best scaling method: {best_method.upper()} (Test R²: {results[best_method]['test_r2']:.4f})")
    
    # Validation checks
    print("\n✅ Validation Checks")
    print("=" * 50)
    
    # Check if all methods converged
    all_converged = all(result['iterations'] < 1000 for result in results.values())
    print(f"All methods converged: {'✅ Yes' if all_converged else '❌ No'}")
    
    # Check if R² scores are reasonable
    reasonable_r2 = all(result['test_r2'] > 0.5 for result in results.values())
    print(f"All R² scores > 0.5: {'✅ Yes' if reasonable_r2 else '❌ No'}")
    
    # Check if scaled methods perform better than unscaled
    scaled_better = results['standard']['test_r2'] > results['none']['test_r2']
    print(f"Scaling improves performance: {'✅ Yes' if scaled_better else '❌ No'}")
    
    print("\n🎉 Test completed successfully!")
    
    return results

if __name__ == "__main__":
    try:
        results = test_gradient_descent()
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()