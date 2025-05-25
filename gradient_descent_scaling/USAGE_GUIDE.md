# 📚 Gradient Descent Scaling - Usage Guide

## 🚀 Quick Start

### 1. Launch the Application
```bash
cd gradient_descent_scaling
streamlit run app.py
```

### 2. Upload Your Dataset
- Click "Choose a CSV or Excel file"
- Supported formats: `.csv`, `.xlsx`, `.xls`
- Your dataset should contain:
  - Numeric features (independent variables)
  - One target variable (dependent variable)
  - At least 50+ samples for meaningful results

### 3. Explore Your Data
The app automatically performs:
- **Data type analysis**
- **Missing value detection**
- **Statistical summaries**
- **Correlation analysis**
- **Data visualization**

### 4. Configure Parameters
Use the sidebar to adjust:
- **Learning Rate** (0.001 - 1.0): Controls step size in optimization
- **Max Iterations** (100 - 5000): Maximum training steps
- **Tolerance** (1e-6 to 1e-3): Convergence threshold
- **Test Size** (0.1 - 0.5): Proportion of data for testing

### 5. Select Target Variable
Choose your dependent variable from the dropdown menu.

### 6. Choose Scaling Methods
Select one or more scaling methods to compare:
- **Standard Scaling**: Best for normally distributed features
- **Min-Max Scaling**: Good when you need bounded values [0,1]
- **Robust Scaling**: Best when data contains outliers
- **No Scaling**: Baseline comparison

### 7. Train and Compare Models
Click "Train Models" to start the optimization process and view results.

## 📊 Understanding the Results

### Performance Metrics Table
- **Train/Test MSE**: Lower is better (measures prediction error)
- **Train/Test R²**: Higher is better (0-1 scale, measures explained variance)
- **Iterations**: Number of steps to convergence

### Cost Function Plots
- **Steep decline**: Good learning rate
- **Oscillations**: Learning rate too high
- **Slow convergence**: Learning rate too low
- **Plateau**: Model converged

### Prediction Plots
- **Points near diagonal**: Good predictions
- **Scattered points**: Poor model fit
- **Systematic bias**: Model underfitting

### Feature Weights
- **Large weights**: Important features
- **Small weights**: Less important features
- **Weight variations**: Impact of scaling

## 🎯 Best Practices

### Data Preparation
1. **Clean your data**: Remove or handle missing values
2. **Feature selection**: Include only relevant features
3. **Sufficient samples**: Use at least 50+ samples
4. **Avoid multicollinearity**: Remove highly correlated features

### Parameter Tuning
1. **Learning Rate**:
   - Start with 0.01
   - Increase if convergence is slow
   - Decrease if cost oscillates or diverges

2. **Max Iterations**:
   - Start with 1000
   - Increase if model hasn't converged
   - Monitor cost function plots

3. **Tolerance**:
   - Use 1e-6 for precise convergence
   - Use 1e-4 for faster training

### Scaling Method Selection
1. **Standard Scaling**: 
   - Default choice for most datasets
   - Works well with normally distributed features

2. **Min-Max Scaling**:
   - Use when features need to be bounded
   - Good for neural networks

3. **Robust Scaling**:
   - Use when data contains outliers
   - Less sensitive to extreme values

4. **No Scaling**:
   - Only when features are already on similar scales
   - Use for comparison purposes

## 🔧 Troubleshooting

### Common Issues

#### 1. Model Not Converging
**Symptoms**: Cost function doesn't decrease or oscillates
**Solutions**:
- Reduce learning rate (try 0.001)
- Increase max iterations
- Try different scaling method
- Check for data quality issues

#### 2. Poor Performance (Low R²)
**Symptoms**: R² score below 0.5
**Solutions**:
- Add more relevant features
- Remove outliers
- Try polynomial features
- Check for data leakage

#### 3. Numerical Instability
**Symptoms**: NaN or infinite values
**Solutions**:
- Use feature scaling
- Reduce learning rate
- Check for extreme values in data
- Normalize target variable

#### 4. Slow Convergence
**Symptoms**: Takes many iterations to converge
**Solutions**:
- Increase learning rate carefully
- Use standard scaling
- Check feature scales

### Error Messages

#### "Input contains NaN"
- Check your dataset for missing values
- Ensure proper data preprocessing
- Verify file format and encoding

#### "Cost became inf"
- Learning rate is too high
- Features need scaling
- Check for extreme outliers

#### "Gradients became NaN/inf"
- Numerical instability detected
- Reduce learning rate
- Apply feature scaling

## 📈 Sample Datasets

The project includes three sample datasets for testing:

### 1. Housing Prices (`housing_prices.csv`)
- **Features**: House size, bedrooms, bathrooms, age, location score
- **Target**: Price
- **Use case**: Real estate price prediction

### 2. Student Performance (`student_performance.csv`)
- **Features**: Study hours, previous score, sleep hours, exercise, stress level
- **Target**: Final exam score
- **Use case**: Academic performance prediction

### 3. Sales Prediction (`sales_prediction.csv`)
- **Features**: Advertising budget, website traffic, social media followers, email subscribers, season
- **Target**: Monthly sales
- **Use case**: Business sales forecasting

## 🎓 Educational Value

### Learning Objectives
1. **Understand gradient descent optimization**
2. **See the impact of feature scaling**
3. **Compare different scaling methods**
4. **Analyze convergence behavior**
5. **Interpret model performance metrics**

### Key Concepts Demonstrated
- **Feature scaling importance**
- **Gradient descent convergence**
- **Numerical stability in optimization**
- **Model evaluation metrics**
- **Hyperparameter tuning**

## 🔬 Advanced Usage

### Custom Datasets
For best results, your dataset should:
- Have 100+ samples
- Include 3-10 relevant features
- Have a continuous target variable
- Be free of extreme outliers
- Have minimal missing values

### Hyperparameter Optimization
1. **Grid Search**: Try different learning rates [0.001, 0.01, 0.1]
2. **Early Stopping**: Monitor validation loss
3. **Learning Rate Scheduling**: Start high, reduce over time

### Feature Engineering
Consider adding:
- **Polynomial features**: x², x³, xy interactions
- **Log transformations**: For skewed features
- **Binning**: Convert continuous to categorical
- **Domain-specific features**: Based on problem context

## 📚 Further Reading

### Gradient Descent Resources
- [Gradient Descent Explained](https://ml-cheatsheet.readthedocs.io/en/latest/gradient_descent.html)
- [Feature Scaling Techniques](https://scikit-learn.org/stable/modules/preprocessing.html)
- [Linear Regression Theory](https://en.wikipedia.org/wiki/Linear_regression)

### Machine Learning Best Practices
- [Model Evaluation Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Cross-Validation Techniques](https://scikit-learn.org/stable/modules/cross_validation.html)
- [Bias-Variance Tradeoff](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff)

## 🤝 Support

If you encounter issues:
1. Check this usage guide
2. Review the troubleshooting section
3. Examine the sample datasets for reference
4. Verify your data format and quality

## 🎉 Success Tips

1. **Start simple**: Use default parameters first
2. **Visualize everything**: Use the provided plots
3. **Compare methods**: Try multiple scaling approaches
4. **Iterate**: Adjust parameters based on results
5. **Understand**: Don't just optimize, learn why it works