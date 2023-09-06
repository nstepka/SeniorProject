import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from collections import defaultdict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Assuming preprocessed_data is already loaded
subsetdf = preprocessed_data.copy()

# Split data into 90/10
X = subsetdf.drop(columns=['price'])
y = subsetdf['price']
X_90, X_test, y_90, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

feature_counts = defaultdict(int)
feature_ranking_sum = defaultdict(int)

# Training and Validation loop
for _ in range(200):  # 200 validation cycles
    # Further split X_90 and y_90 into 70/30 for training and validation
    X_train, X_val, y_train, y_val = train_test_split(X_90, y_90, test_size=0.3, random_state=42)

    for _ in range(1000):  # 1000 training cycles
        # Randomly select 15 features excluding 'price'
        chosen_features = np.random.choice(X_train.columns, 15, replace=False)
        
        X_train_subset = X_train[chosen_features]
        model = GradientBoostingRegressor()
        model.fit(X_train_subset, y_train)

        # Update feature importance counts and rankings
        feature_importances = model.feature_importances_
        ranked_features = sorted(zip(chosen_features, feature_importances), key=lambda x: x[1], reverse=True)
        
        for rank, (feature_name, _) in enumerate(ranked_features):
            feature_counts[feature_name] += 1
            feature_ranking_sum[feature_name] += (rank + 1)  # rank is 0-indexed

# Calculate average ranking for features
average_ranking = {feature: total_rank / feature_counts[feature] for feature, total_rank in feature_ranking_sum.items()}

# Sort features by their average ranking
sorted_features = sorted(average_ranking.items(), key=lambda x: x[1])

# Test the final model on the 10% test set using all features
model = GradientBoostingRegressor()
model.fit(X_90, y_90)
y_test_pred = model.predict(X_test)

# Creating the summary using pandas dataframe
summary_df = pd.DataFrame({
    'Feature': [x[0] for x in sorted_features],
    'Count': [feature_counts[x[0]] for x in sorted_features],
    'Average_Ranking': [x[1] for x in sorted_features]
})

print("Feature Importance Summary:")
print(summary_df.head(15))

print("\nValidation Performance (R2, MSE, MAE):")
print([r2_score(y_val, model.predict(X_val)), mean_squared_error(y_val, model.predict(X_val)), mean_absolute_error(y_val, model.predict(X_val))])

print("\nTest Performance (R2, MSE, MAE):")
print([r2_score(y_test, y_test_pred), mean_squared_error(y_test, y_test_pred), mean_absolute_error(y_test, y_test_pred)])
