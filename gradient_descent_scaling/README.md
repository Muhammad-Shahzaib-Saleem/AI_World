# 📈 Gradient Descent with Feature Scaling

A comprehensive Streamlit application that demonstrates the impact of different feature scaling methods on gradient descent optimization for regression tasks.

## 🌟 Features

### 📊 Data Analysis & Preprocessing
- **File Upload**: Support for CSV and Excel files
- **Exploratory Data Analysis**: Statistical summaries, correlation matrices, missing value analysis
- **Automatic Preprocessing**: Missing value handling, categorical encoding, data normalization
- **Interactive Data Visualization**: Heatmaps, correlation plots, and statistical charts

### 🔄 Feature Scaling Methods
- **Standard Scaling (Z-score)**: Normalizes features to have mean=0 and std=1
- **Min-Max Scaling**: Scales features to a fixed range [0,1]
- **Robust Scaling**: Uses median and IQR, robust to outliers
- **No Scaling**: Baseline comparison without any scaling

### 🤖 Custom Gradient Descent Implementation
- **Configurable Parameters**: Learning rate, max iterations, tolerance
- **Cost Function Tracking**: Real-time monitoring of MSE convergence
- **Multiple Scaling Comparison**: Side-by-side comparison of different scaling methods
- **Performance Metrics**: MSE, RMSE, MAE, R² score for train and test sets

### 📈 Advanced Visualizations
- **Cost Function Convergence**: Interactive plots showing optimization progress
- **Prediction vs Actual**: Scatter plots comparing model predictions
- **Feature Weights**: Bar charts showing feature importance across scaling methods
- **Performance Comparison**: Comprehensive metrics table

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd gradient_descent_scaling
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

4. **Access the app**: Open your browser and go to `http://localhost:8501`

### Usage

1. **Upload Dataset**: 
   - Click "Choose a CSV or Excel file"
   - Upload your regression dataset
   - Ensure it contains numeric features and a target variable

2. **Explore Data**:
   - Review the exploratory data analysis
   - Check data types, missing values, and correlations
   - Examine statistical summaries

3. **Configure Parameters**:
   - Adjust learning rate (0.001 - 1.0)
   - Set maximum iterations (100 - 5000)
   - Choose convergence tolerance
   - Select train/test split ratio

4. **Select Target Variable**:
   - Choose the dependent variable for regression
   - The app will automatically use remaining columns as features

5. **Choose Scaling Methods**:
   - Select one or more scaling methods to compare
   - Standard and Min-Max scaling are selected by default

6. **Train Models**:
   - Click "Train Models" to start the gradient descent optimization
   - Monitor progress and convergence for each scaling method

7. **Analyze Results**:
   - Compare performance metrics across scaling methods
   - Examine cost function convergence plots
   - Review prediction accuracy visualizations
   - Analyze feature weights and importance

## 📊 Sample Datasets

The application works best with:
- **Regression datasets** with continuous target variables
- **Numeric features** (categorical variables are automatically encoded)
- **Clean or moderately messy data** (missing values are handled automatically)

### Recommended Test Datasets:
- Boston Housing Dataset
- California Housing Dataset
- Wine Quality Dataset
- Any CSV with numeric features and a continuous target

## 🔧 Technical Details

### Gradient Descent Algorithm

The custom implementation includes:

```python
# Cost Function (Mean Squared Error)
J(θ) = (1/2m) * Σ(hθ(x) - y)²

# Gradient Computation
∂J/∂θ = (1/m) * X^T * (Xθ - y)

# Parameter Update
θ = θ - α * ∂J/∂θ
```

### Feature Scaling Methods

1. **Standard Scaling**:
   ```
   x_scaled = (x - μ) / σ
   ```

2. **Min-Max Scaling**:
   ```
   x_scaled = (x - min) / (max - min)
   ```

3. **Robust Scaling**:
   ```
   x_scaled = (x - median) / IQR
   ```

### Performance Metrics

- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of Determination

## 🎯 Key Benefits

### Educational Value
- **Visual Learning**: See how scaling affects convergence
- **Interactive Exploration**: Experiment with different parameters
- **Comparative Analysis**: Side-by-side method comparison

### Practical Applications
- **Data Science Workflows**: Understand preprocessing impact
- **Model Optimization**: Choose optimal scaling methods
- **Feature Engineering**: Analyze feature importance

### Performance Insights
- **Convergence Speed**: Compare iteration requirements
- **Numerical Stability**: Observe gradient descent behavior
- **Generalization**: Evaluate test set performance

## 🔍 Understanding the Results

### Cost Function Plots
- **Steep Initial Decline**: Good learning rate
- **Oscillations**: Learning rate might be too high
- **Slow Convergence**: Learning rate might be too low
- **Plateau**: Model has converged

### Prediction Plots
- **Points Near Diagonal**: Good predictions
- **Scattered Points**: Poor model fit
- **Systematic Bias**: Model underfitting

### Feature Weights
- **Large Weights**: Important features
- **Small Weights**: Less important features
- **Weight Variations**: Impact of scaling on feature importance

## 🛠️ Customization

### Adding New Scaling Methods
```python
def custom_scaler(X):
    # Implement your scaling logic
    return X_scaled

# Add to scaling_methods dictionary
scaling_methods['custom'] = 'Custom Scaling'
```

### Modifying Gradient Descent
```python
# Adjust the GradientDescentRegressor class
# Add momentum, adaptive learning rates, etc.
```

## 📝 Best Practices

### Data Preparation
- **Clean Data**: Handle missing values appropriately
- **Feature Selection**: Remove irrelevant features
- **Outlier Detection**: Consider robust scaling for outlier-heavy data

### Parameter Tuning
- **Learning Rate**: Start with 0.01, adjust based on convergence
- **Iterations**: Increase if model hasn't converged
- **Tolerance**: Use smaller values for precise convergence

### Scaling Method Selection
- **Standard Scaling**: Good default choice
- **Min-Max Scaling**: When you need bounded features
- **Robust Scaling**: When data contains outliers
- **No Scaling**: Only when features are already on similar scales

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:
- New scaling methods
- Additional visualizations
- Performance improvements
- Bug fixes

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Built with Streamlit for interactive web applications
- Uses scikit-learn for preprocessing utilities
- Plotly for advanced interactive visualizations
- Pandas and NumPy for data manipulation