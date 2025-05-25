import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class GradientDescentRegressor:
    """Custom Gradient Descent implementation with different scaling algorithms"""
    
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6, scaling_method='standard', adaptive_lr=True):
        self.learning_rate = learning_rate
        self.initial_learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.scaling_method = scaling_method
        self.adaptive_lr = adaptive_lr
        self.weights = None
        self.bias = None
        self.cost_history = []
        self.scaler = None
        
    def _initialize_scaler(self):
        """Initialize the appropriate scaler based on scaling method"""
        if self.scaling_method == 'standard':
            return StandardScaler()
        elif self.scaling_method == 'minmax':
            return MinMaxScaler()
        elif self.scaling_method == 'robust':
            return RobustScaler()
        else:
            return None
    
    def _scale_features(self, X, fit=True):
        """Scale features using the selected scaling method"""
        if self.scaling_method == 'none':
            return X
        
        if fit:
            self.scaler = self._initialize_scaler()
            return self.scaler.fit_transform(X)
        else:
            return self.scaler.transform(X)
    
    def _compute_cost(self, X, y):
        """Compute the mean squared error cost"""
        m = len(y)
        predictions = X.dot(self.weights) + self.bias
        cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
        return cost
    
    def _compute_gradients(self, X, y):
        """Compute gradients for weights and bias"""
        m = len(y)
        predictions = X.dot(self.weights) + self.bias
        
        dw = (1 / m) * X.T.dot(predictions - y)
        db = (1 / m) * np.sum(predictions - y)
        
        return dw, db
    
    def fit(self, X, y):
        """Train the model using gradient descent"""
        # Convert to numpy arrays if needed
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
            
        # Scale features
        X_scaled = self._scale_features(X, fit=True)
        
        # Initialize parameters
        m, n = X_scaled.shape
        self.weights = np.random.normal(0, 0.01, n)
        self.bias = 0
        self.cost_history = []
        
        # Normalize target variable for better numerical stability
        y_mean = np.mean(y)
        y_std = np.std(y)
        y_normalized = (y - y_mean) / y_std
        
        # Store normalization parameters
        self.y_mean = y_mean
        self.y_std = y_std
        
        # Store detailed iteration information
        self.iteration_details = []
        
        # Gradient descent
        for i in range(self.max_iterations):
            # Compute cost
            cost = self._compute_cost(X_scaled, y_normalized)
            self.cost_history.append(cost)
            
            # Store iteration details
            iteration_info = {
                'iteration': i + 1,
                'cost': cost,
                'learning_rate': self.learning_rate,
                'weights': self.weights.copy(),
                'bias': self.bias
            }
            self.iteration_details.append(iteration_info)
            
            # Check for NaN or infinite values
            if np.isnan(cost) or np.isinf(cost):
                print(f"Warning: Cost became {cost} at iteration {i}")
                break
            
            # Adaptive learning rate: reduce if cost is increasing
            if self.adaptive_lr and i > 0 and cost > self.cost_history[-2]:
                self.learning_rate *= 0.5
                if self.learning_rate < 1e-8:
                    print(f"Learning rate became too small at iteration {i}")
                    break
            
            # Compute gradients
            dw, db = self._compute_gradients(X_scaled, y_normalized)
            
            # Check for NaN or infinite gradients
            if np.any(np.isnan(dw)) or np.any(np.isinf(dw)) or np.isnan(db) or np.isinf(db):
                print(f"Warning: Gradients became NaN/inf at iteration {i}")
                break
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Check for convergence
            if i > 0 and abs(self.cost_history[-2] - self.cost_history[-1]) < self.tolerance:
                if 'st' in globals():
                    st.info(f"Converged after {i+1} iterations")
                break
        
        return self
    
    def predict(self, X):
        """Make predictions on new data"""
        # Convert to numpy array if needed
        if hasattr(X, 'values'):
            X = X.values
            
        X_scaled = self._scale_features(X, fit=False)
        predictions_normalized = X_scaled.dot(self.weights) + self.bias
        
        # Denormalize predictions
        predictions = predictions_normalized * self.y_std + self.y_mean
        return predictions
    
    def get_metrics(self, X, y):
        """Calculate performance metrics"""
        predictions = self.predict(X)
        mse = mean_squared_error(y, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, predictions)
        r2 = r2_score(y, predictions)
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2
        }
    
    def get_iteration_details(self):
        """Get detailed information about each iteration"""
        return self.iteration_details

def load_and_preprocess_data(uploaded_file):
    """Load and preprocess the uploaded dataset"""
    try:
        # Read the file
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload CSV or Excel files.")
            return None
        
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def perform_eda(df):
    """Perform Exploratory Data Analysis"""
    st.subheader("📊 Exploratory Data Analysis")
    
    # Basic info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", df.shape[0])
    with col2:
        st.metric("Columns", df.shape[1])
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    
    # Data types
    st.subheader("Data Types")
    st.dataframe(df.dtypes.to_frame('Data Type'))
    
    # Missing values heatmap
    if df.isnull().sum().sum() > 0:
        st.subheader("Missing Values Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.isnull(), cbar=True, ax=ax)
        st.pyplot(fig)
    
    # Statistical summary
    st.subheader("Statistical Summary")
    st.dataframe(df.describe())
    
    # Correlation matrix for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        st.subheader("Correlation Matrix")
        fig, ax = plt.subplots(figsize=(10, 8))
        correlation_matrix = df[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
        st.pyplot(fig)

def preprocess_dataset(df):
    """Preprocess the dataset for machine learning"""
    st.subheader("🔧 Data Preprocessing")
    
    # Handle missing values
    if df.isnull().sum().sum() > 0:
        st.warning("Missing values detected. Handling missing values...")
        
        # For numeric columns, fill with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        # For categorical columns, fill with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        st.success("Missing values handled successfully!")
    
    # Encode categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        st.info(f"Encoding categorical variables: {list(categorical_cols)}")
        df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    else:
        df_encoded = df.copy()
    
    # Remove any remaining non-numeric columns
    df_numeric = df_encoded.select_dtypes(include=[np.number])
    
    return df_numeric

def create_cost_visualization(cost_history, scaling_method):
    """Create cost function visualization"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(cost_history))),
        y=cost_history,
        mode='lines',
        name=f'Cost ({scaling_method})',
        line=dict(width=2)
    ))
    
    fig.update_layout(
        title=f'Cost Function Convergence - {scaling_method.title()} Scaling',
        xaxis_title='Iterations',
        yaxis_title='Cost (MSE)',
        template='plotly_white'
    )
    
    return fig

def create_prediction_plot(y_true, y_pred, title):
    """Create prediction vs actual plot"""
    fig = go.Figure()
    
    # Perfect prediction line
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(dash='dash', color='red')
    ))
    
    # Actual vs predicted
    fig.add_trace(go.Scatter(
        x=y_true,
        y=y_pred,
        mode='markers',
        name='Predictions',
        marker=dict(size=6, opacity=0.6)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Actual Values',
        yaxis_title='Predicted Values',
        template='plotly_white'
    )
    
    return fig

def main():
    st.set_page_config(
        page_title="Gradient Descent Scaling",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("📈 Gradient Descent with Feature Scaling")
    st.markdown("Upload your dataset and compare different scaling methods for gradient descent optimization")
    
    # Sidebar for parameters
    st.sidebar.header("⚙️ Model Parameters")
    learning_rate = st.sidebar.slider("Learning Rate", 0.001, 1.0, 0.01, 0.001)
    max_iterations = st.sidebar.slider("Max Iterations", 100, 5000, 1000, 100)
    tolerance = st.sidebar.selectbox("Tolerance", [1e-6, 1e-5, 1e-4, 1e-3], index=0)
    test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
    
    # File upload
    st.header("📁 Upload Dataset")
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload a dataset with numeric features for regression analysis"
    )
    
    if uploaded_file is not None:
        # Load data
        df = load_and_preprocess_data(uploaded_file)
        
        if df is not None:
            st.success(f"✅ Dataset loaded successfully! Shape: {df.shape}")
            
            # Show raw data
            with st.expander("📋 View Raw Data"):
                st.dataframe(df.head(10))
            
            # EDA
            perform_eda(df)
            
            # Preprocess data
            df_processed = preprocess_dataset(df)
            
            # Target variable selection
            st.header("🎯 Target Variable Selection")
            numeric_columns = df_processed.columns.tolist()
            target_column = st.selectbox(
                "Select target variable (dependent variable)",
                numeric_columns,
                index=len(numeric_columns)-1
            )
            
            if target_column:
                # Prepare features and target
                X = df_processed.drop(columns=[target_column])
                y = df_processed[target_column].values
                
                st.info(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
                
                # Train-test split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
                
                # Scaling methods comparison
                st.header("🔄 Scaling Methods Comparison")
                
                scaling_methods = {
                    'none': 'No Scaling',
                    'standard': 'Standard Scaling (Z-score)',
                    'minmax': 'Min-Max Scaling',
                    'robust': 'Robust Scaling'
                }
                
                selected_methods = st.multiselect(
                    "Select scaling methods to compare",
                    list(scaling_methods.keys()),
                    default=['standard', 'minmax'],
                    format_func=lambda x: scaling_methods[x]
                )
                
                if st.button("🚀 Train Models", type="primary"):
                    results = {}
                    
                    # Progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, method in enumerate(selected_methods):
                        status_text.text(f"Training with {scaling_methods[method]}...")
                        
                        # Train model
                        model = GradientDescentRegressor(
                            learning_rate=learning_rate,
                            max_iterations=max_iterations,
                            tolerance=tolerance,
                            scaling_method=method
                        )
                        
                        model.fit(X_train, y_train)
                        
                        # Get predictions and metrics
                        train_metrics = model.get_metrics(X_train, y_train)
                        test_metrics = model.get_metrics(X_test, y_test)
                        
                        results[method] = {
                            'model': model,
                            'train_metrics': train_metrics,
                            'test_metrics': test_metrics,
                            'train_predictions': model.predict(X_train),
                            'test_predictions': model.predict(X_test)
                        }
                        
                        progress_bar.progress((i + 1) / len(selected_methods))
                    
                    status_text.text("Training completed!")
                    
                    # Display results
                    st.header("📊 Results Comparison")
                    
                    # Metrics comparison table
                    st.subheader("Performance Metrics")
                    metrics_df = []
                    for method, result in results.items():
                        row = {
                            'Scaling Method': scaling_methods[method],
                            'Train MSE': f"{result['train_metrics']['MSE']:.4f}",
                            'Test MSE': f"{result['test_metrics']['MSE']:.4f}",
                            'Train R²': f"{result['train_metrics']['R²']:.4f}",
                            'Test R²': f"{result['test_metrics']['R²']:.4f}",
                            'Iterations': len(result['model'].cost_history)
                        }
                        metrics_df.append(row)
                    
                    metrics_df = pd.DataFrame(metrics_df)
                    st.dataframe(metrics_df, use_container_width=True)
                    
                    # Cost function plots
                    st.subheader("Cost Function Convergence")
                    cost_cols = st.columns(len(selected_methods))
                    
                    for i, (method, result) in enumerate(results.items()):
                        with cost_cols[i]:
                            fig = create_cost_visualization(
                                result['model'].cost_history, 
                                scaling_methods[method]
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Prediction plots
                    st.subheader("Predictions vs Actual Values")
                    pred_cols = st.columns(len(selected_methods))
                    
                    for i, (method, result) in enumerate(results.items()):
                        with pred_cols[i]:
                            fig = create_prediction_plot(
                                y_test,
                                result['test_predictions'],
                                f"{scaling_methods[method]} - Test Set"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Feature importance (weights visualization)
                    st.subheader("Feature Weights Comparison")
                    weights_data = []
                    feature_names = X.columns.tolist()
                    
                    for method, result in results.items():
                        for i, weight in enumerate(result['model'].weights):
                            weights_data.append({
                                'Feature': feature_names[i],
                                'Weight': weight,
                                'Scaling Method': scaling_methods[method]
                            })
                    
                    weights_df = pd.DataFrame(weights_data)
                    
                    fig = px.bar(
                        weights_df,
                        x='Feature',
                        y='Weight',
                        color='Scaling Method',
                        barmode='group',
                        title='Feature Weights by Scaling Method'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Iteration Details Section
                    st.subheader("🔍 Detailed Iteration Analysis")
                    
                    # Method selection for iteration details
                    selected_method_for_details = st.selectbox(
                        "Select scaling method to view iteration details:",
                        options=list(results.keys()),
                        format_func=lambda x: scaling_methods[x],
                        key="iteration_details_method"
                    )
                    
                    if selected_method_for_details in results:
                        model = results[selected_method_for_details]['model']
                        iteration_details = model.get_iteration_details()
                        
                        # Create tabs for different views
                        tab1, tab2, tab3 = st.tabs(["📊 Summary Table", "📈 Cost Progress", "⚙️ Parameters Evolution"])
                        
                        with tab1:
                            st.write(f"**Iteration Details for {scaling_methods[selected_method_for_details]}**")
                            
                            # Show iteration summary (every 10th iteration for large datasets)
                            if len(iteration_details) > 50:
                                step = max(1, len(iteration_details) // 50)
                                display_iterations = iteration_details[::step] + [iteration_details[-1]]
                                st.info(f"Showing every {step} iterations for better readability (total: {len(iteration_details)} iterations)")
                            else:
                                display_iterations = iteration_details
                            
                            # Create summary dataframe
                            summary_data = []
                            for detail in display_iterations:
                                summary_data.append({
                                    'Iteration': detail['iteration'],
                                    'Cost': f"{detail['cost']:.6f}",
                                    'Learning Rate': f"{detail['learning_rate']:.6f}",
                                    'Bias': f"{detail['bias']:.6f}",
                                    'Avg Weight': f"{np.mean(np.abs(detail['weights'])):.6f}"
                                })
                            
                            summary_df = pd.DataFrame(summary_data)
                            st.dataframe(summary_df, use_container_width=True, height=400)
                            
                            # Download option for full iteration details
                            if st.button("📥 Download Full Iteration Details", key="download_iterations"):
                                full_data = []
                                for detail in iteration_details:
                                    row = {
                                        'Iteration': detail['iteration'],
                                        'Cost': detail['cost'],
                                        'Learning_Rate': detail['learning_rate'],
                                        'Bias': detail['bias']
                                    }
                                    # Add individual weights
                                    for i, weight in enumerate(detail['weights']):
                                        row[f'Weight_{i+1}'] = weight
                                    full_data.append(row)
                                
                                full_df = pd.DataFrame(full_data)
                                csv = full_df.to_csv(index=False)
                                st.download_button(
                                    label="Download CSV",
                                    data=csv,
                                    file_name=f"iteration_details_{selected_method_for_details}.csv",
                                    mime="text/csv"
                                )
                        
                        with tab2:
                            st.write("**Cost Function Progress Over Iterations**")
                            
                            # Create detailed cost plot
                            iterations = [d['iteration'] for d in iteration_details]
                            costs = [d['cost'] for d in iteration_details]
                            
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=iterations,
                                y=costs,
                                mode='lines+markers',
                                name='Cost',
                                line=dict(color='blue', width=2),
                                marker=dict(size=4)
                            ))
                            
                            fig.update_layout(
                                title=f'Cost Function Convergence - {scaling_methods[selected_method_for_details]}',
                                xaxis_title='Iteration',
                                yaxis_title='Cost (MSE)',
                                hovermode='x unified',
                                height=400
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Show convergence statistics
                            if len(costs) > 1:
                                initial_cost = costs[0]
                                final_cost = costs[-1]
                                cost_reduction = ((initial_cost - final_cost) / initial_cost) * 100
                                
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Initial Cost", f"{initial_cost:.6f}")
                                with col2:
                                    st.metric("Final Cost", f"{final_cost:.6f}")
                                with col3:
                                    st.metric("Cost Reduction", f"{cost_reduction:.2f}%")
                                with col4:
                                    st.metric("Total Iterations", len(iteration_details))
                        
                        with tab3:
                            st.write("**Parameters Evolution Over Iterations**")
                            
                            # Plot learning rate evolution
                            learning_rates = [d['learning_rate'] for d in iteration_details]
                            
                            fig_lr = go.Figure()
                            fig_lr.add_trace(go.Scatter(
                                x=iterations,
                                y=learning_rates,
                                mode='lines+markers',
                                name='Learning Rate',
                                line=dict(color='red', width=2),
                                marker=dict(size=4)
                            ))
                            
                            fig_lr.update_layout(
                                title='Learning Rate Evolution',
                                xaxis_title='Iteration',
                                yaxis_title='Learning Rate',
                                height=300
                            )
                            
                            st.plotly_chart(fig_lr, use_container_width=True)
                            
                            # Plot weights evolution
                            if len(iteration_details) > 0:
                                n_weights = len(iteration_details[0]['weights'])
                                
                                fig_weights = go.Figure()
                                
                                for i in range(min(n_weights, 5)):  # Show max 5 weights for clarity
                                    weight_values = [d['weights'][i] for d in iteration_details]
                                    fig_weights.add_trace(go.Scatter(
                                        x=iterations,
                                        y=weight_values,
                                        mode='lines',
                                        name=f'Weight {i+1}',
                                        line=dict(width=2)
                                    ))
                                
                                fig_weights.update_layout(
                                    title='Weights Evolution (First 5 Features)',
                                    xaxis_title='Iteration',
                                    yaxis_title='Weight Value',
                                    height=300
                                )
                                
                                st.plotly_chart(fig_weights, use_container_width=True)
                                
                                if n_weights > 5:
                                    st.info(f"Showing first 5 out of {n_weights} feature weights for clarity")
                    
                    # Best model recommendation
                    st.subheader("🏆 Best Model Recommendation")
                    best_method = min(results.keys(), 
                                    key=lambda x: results[x]['test_metrics']['MSE'])
                    best_result = results[best_method]
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "Best Scaling Method",
                            scaling_methods[best_method]
                        )
                    with col2:
                        st.metric(
                            "Test MSE",
                            f"{best_result['test_metrics']['MSE']:.4f}"
                        )
                    with col3:
                        st.metric(
                            "Test R²",
                            f"{best_result['test_metrics']['R²']:.4f}"
                        )
                    
                    st.success(f"🎉 Best performing model uses {scaling_methods[best_method]} with Test R² of {best_result['test_metrics']['R²']:.4f}")
    
    # Information section
    st.markdown("---")
    with st.expander("ℹ️ About Feature Scaling in Gradient Descent"):
        st.markdown("""
        **Feature Scaling** is crucial for gradient descent optimization:
        
        **Why Scaling Matters:**
        - Features with different scales can dominate the cost function
        - Gradient descent converges faster with scaled features
        - Prevents numerical instability
        
        **Scaling Methods:**
        - **Standard Scaling**: (x - μ) / σ (mean=0, std=1)
        - **Min-Max Scaling**: (x - min) / (max - min) (range 0-1)
        - **Robust Scaling**: Uses median and IQR (robust to outliers)
        
        **Gradient Descent Algorithm:**
        1. Initialize weights randomly
        2. Calculate cost (MSE)
        3. Compute gradients
        4. Update weights: w = w - α∇J(w)
        5. Repeat until convergence
        """)

if __name__ == "__main__":
    main()