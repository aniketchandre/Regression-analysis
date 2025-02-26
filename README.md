# Regression-analysis
# Step 1: Load and Explore the Dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv('house_prices.csv')

# Display the first few rows
print("First few rows of the dataset:")
print(df.head())

# Check for missing values
print("\nMissing values in the dataset:")
print(df.isnull().sum())

# Analyze distributions of numerical variables
print("\nDistribution of Size:")
sns.histplot(df['Size'], kde=True)
plt.show()

print("\nDistribution of Price:")
sns.histplot(df['Price'], kde=True)
plt.show()

# Identify potential outliers
print("\nBoxplot for Size:")
sns.boxplot(x=df['Size'])
plt.show()

print("\nBoxplot for Price:")
sns.boxplot(x=df['Price'])
plt.show()

# Step 2: Data Preprocessing
# Normalize Numerical Data
scaler = MinMaxScaler()
df[['Size', 'Number of Rooms']] = scaler.fit_transform(df[['Size', 'Number of Rooms']])

# Encode Categorical Features
df = pd.get_dummies(df, columns=['Location'], drop_first=True)

# Step 3: Feature Selection
# Correlation analysis
print("\nCorrelation matrix with Price:")
correlation_matrix = df.corr()
print(correlation_matrix['Price'].sort_values(ascending=False))

# Step 4: Model Training
# Define features and target variable
X = df.drop('Price', axis=1)
y = df['Price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Model Evaluation
# Predict on the test set
y_pred = model.predict(X_test)

# Calculate RMSE
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f'\nRMSE: {rmse}')

# Calculate R²
r2 = r2_score(y_test, y_pred)
print(f'R²: {r2}')

# Deliverables
# 1. Trained Regression Model
# The fitted linear regression model is stored in the `model` variable.

# 2. Predictions
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print("\nActual vs Predicted Prices:")
print(results.head())

# 3. Evaluation Metrics
# RMSE and R² values are printed during the evaluation step.

# 4. Feature Insights
# The correlation matrix provides insights into the most important predictors influencing house prices.
print("\nFeature Insights:")
print(correlation_matrix['Price'].sort_values(ascending=False))
