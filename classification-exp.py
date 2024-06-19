from metrics import *
from tree.utils import *
from tree.base import *
import numpy as np
from sklearn.datasets import make_classification
import pandas as pd
from sklearn.model_selection import train_test_split
X, y = make_classification(
n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

# For plotting
import matplotlib.pyplot as plt
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

def kfold_validation(data, k):
    fold_size = len(data) // k
    indices = np.arange(len(data))
    folds = []
    for i in range(k):
        test_indices = indices[i * fold_size: (i + 1) * fold_size]
        train_indices = np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]])
        folds.append((train_indices, test_indices))
    return folds

# Define the number of folds (K)
k = 5

# Get the fold indices
fold_indices = kfold_validation(X, k)

model = DecisionTree()

# Initialize a list to store the evaluation scores
scores = []

# Iterate through each fold
for train_indices, test_indices in fold_indices:
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    
    X_train=pd.DataFrame(X_train)
    y_train=pd.Series(y_train)
    X_test=pd.DataFrame(X_test)
    y_test=pd.Series(y_test)

    # Train the model on the training data
    model.fit(X_train, y_train)
    
    # Make predictions on the test data
    y_pred = model.predict(X_test)
    
    # Calculate the accuracy score for this fold
    fold_score = accuracy(y_test, y_pred)
    
    # Append the fold score to the list of scores
    scores.append(fold_score)

# Calculate the mean accuracy across all folds
mean_accuracy = np.mean(scores)
print("K-Fold Cross-Validation Scores:", scores)
print("Mean Accuracy:", mean_accuracy)

#cross-validation done

arr = X

features = pd.DataFrame(arr, columns=["Feature 1", "Feature 2"],dtype='float')
target = pd.DataFrame(y, columns=['Target'], dtype='category')

print(features)
print(target)
print(target.dtypes)
attributes_train, attributes_test, target_train, target_test=train_test_split(features, target, random_state = 42, train_size = 0.7)
print(type(attributes_train))

#re-indexing the dataframes to avoid index errors
attributes_train.reset_index(drop=True, inplace=True)
attributes_test.reset_index(drop=True, inplace=True)
target_train.reset_index(drop=True, inplace=True)
target_test.reset_index(drop=True, inplace=True)

#converting the target train and test single-columned dataframes to series
target_train_ser = target_train.squeeze()
target_test_ser = target_test.squeeze()


#without optimal hyperparameters:
dt=DecisionTree()
dt.fit(attributes_train, target_train_ser)
pred=dt.predict(attributes_test)
print(pred)

accuracy1=accuracy(pred, target_test_ser)

print(accuracy1)
for cls in target.Target.unique():
  print("precision", precision(pred, target_test_ser, cls ))
  print("recall", recall(pred, target_test_ser, cls ))

#nested cross-validation to find the optimal hyperparameters

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

X_train=pd.DataFrame(X_train)
y_train=pd.Series(y_train)
X_test=pd.DataFrame(X_test)
y_test=pd.Series(y_test)
X_val=pd.DataFrame(X_val)
y_val=pd.Series(y_val)

print("Training samples: {}".format(len(X_train)))
print("Validation samples: {}".format(len(X_val)))
print("Testing samples: {}".format(len(X_test)))

hyperparameters = {}
hyperparameters['max_depth'] = [1,2,3,4,5,6,7,8,9,10]
hyperparameters['criteria_values'] = ['information_gain', 'gini_index']

best_accuracy = 0
best_hyperparameters = {}

output = {}
count = 0

#nested cross-validation:
for max_depth in hyperparameters['max_depth']:
        for criterion in hyperparameters['criteria_values']:
            # Create and fit the decision tree classifier with the current hyperparameters
            dt = DecisionTree(max_depth=max_depth, criterion=criterion)
            dt.fit(X_train, y_train)
            
            # Evaluate the performance on the validation set
            y_hat=dt.predict(X_val)
            val_accuracy = accuracy(y_hat, y_val)
            output[count] = {'max_depth': max_depth, 'criterion': criterion, 'val_accuracy': val_accuracy}
            count += 1

hyperparameters_df = pd.DataFrame(output).T
hyperparameters_df.sort_values(by='val_accuracy', ascending=False)
print(hyperparameters_df)

max_acc=hyperparameters_df['val_accuracy'].max()
print("Maximum Accuracy=",max_acc)
for i in range (len(hyperparameters_df)):
  if hyperparameters_df['val_accuracy'][i]==max_acc:
     print("Optimal depth=",hyperparameters_df['max_depth'][i])
     break

#with optimal hyperparamaters:
print("Now, using these optimal hyperparameters on the sample data and comparing the accuracies: ")
dt=DecisionTree(max_depth=2)
dt.fit(attributes_train, target_train_ser)
pred=dt.predict(attributes_test)
print(pred)

accuracy3=accuracy(pred, target_test_ser)

print(accuracy3)
for cls in target.Target.unique():
  print("precision", precision(pred, target_test_ser, cls ))
  print("recall", recall(pred, target_test_ser, cls ))


#without hyperparameters:
dt=DecisionTree()
dt.fit(attributes_train, target_train_ser)
pred=dt.predict(attributes_test)
print(pred)

accuracy2=accuracy(pred, target_test_ser)

print(accuracy2)
for cls in target.Target.unique():
  print("precision", precision(pred, target_test_ser, cls ))
  print("recall", recall(pred, target_test_ser, cls ))