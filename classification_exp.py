import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Tree.Base import DecisionTree
from metrics import precision as precision_metric, recall as recall_metric, accuracy as accuracy_metric
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

np.random.seed(42)

# Generate dataset
X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

# For plotting
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

# Convert to DataFrame for compatibility
X_df = pd.DataFrame(X, columns=['Feature 1', 'Feature 2'])
y_series = pd.Series(y, name='Target')

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_df, y_series, test_size=0.3, random_state=42)

# Initialize and fit the decision tree
tree = DecisionTree(criterion='information_gain', max_depth=3)
tree.fit(X_train, y_train)

# Predict
y_pred = tree.predict(X_test)

# Calculate metrics for class 0 and class 1
for cls in y_test.unique():
    precision_score = precision_metric(y_test, y_pred, cls)
    recall_score = recall_metric(y_test, y_pred, cls)
    print(f'Precision for class {cls}: {precision_score:.4f}')
    print(f'Recall for class {cls}: {recall_score:.4f}')

# Calculate overall accuracy
accuracy = accuracy_metric(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

# Plot decision boundaries
x_min, x_max = X_test['Feature 1'].min() - 1, X_test['Feature 1'].max() + 1
y_min, y_max = X_test['Feature 2'].min() - 1, X_test['Feature 2'].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

Z = tree.predict(pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=['Feature 1', 'Feature 2']))
Z = Z.values.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(X_test['Feature 1'], X_test['Feature 2'], c=y_test, edgecolors='k', marker='o')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Tree Decision Boundaries')
plt.show()


