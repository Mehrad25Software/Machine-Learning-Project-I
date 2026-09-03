# import both numpy and pandas as we need them for this project
import numpy as np
import pandas as pd

#Loading the data
train_path = "/kaggle/input/comp-432-dataset/train.csv"
test_path = "/kaggle/input/comp-432-dataset/test.csv"


# Entropy function

def entropy(y):
    #if all labels in y are the same then entropy is 0
    if len(y) == 0:
        return 0
    classes, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    # entropy formula
    return -np.sum(p * np.log2(p + 1e-9)) #avoids log(0) logical error


# Tree Node

class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature #index of feature to split on
        self.threshold = threshold #value of the feature to split at
        self.left = left #left child node
        self.right = right #right child node
        self.value = value #predicted clas label or the leaf


# Decision Tree Classifier (NumPy only)

class DecisionTree:
    def __init__(self, max_depth=12, min_samples_split=200):
        self.max_depth = max_depth    #stop when depth reaches this
        self.min_samples_split = min_samples_split   
        self.root = None   # this will hold the root TreeNode

    def fit(self, X, y):
        #start bulding the tree from the root
        self.root = self._grow(X, y, depth=0)

    def _grow(self, X, y, depth):
        num_samples = len(y)
        num_classes = len(np.unique(y))

        # Stopping conditions
        if (depth >= self.max_depth 
            or num_classes == 1
            or num_samples < self.min_samples_split):
            leaf_value = self._most_common(y)
            return TreeNode(value=leaf_value)

        # Best split
        feature, threshold = self._best_split(X, y)
        if feature is None:  # no valid split
            return TreeNode(value=self._most_common(y))
            
        # Split X, y into left and right child sets
        left_idx = X[:, feature] <= threshold
        right_idx = ~left_idx

        left = self._grow(X[left_idx], y[left_idx], depth + 1)
        right = self._grow(X[right_idx], y[right_idx], depth + 1)

        # at this point return an internal node
        return TreeNode(feature, threshold, left, right)

    def _best_split(self, X, y):
        best_gain = 0
        best_feature = None
        best_threshold = None

        parent_entropy = entropy(y)
        n_features = X.shape[1]

        #A for loop to try every feature
        for f in range(n_features):
            thresholds = np.unique(X[:, f])

            for thr in thresholds:
                left_idx = X[:, f] <= thr
                right_idx = ~left_idx

                # skip useless splits
                if left_idx.sum() == 0 or right_idx.sum() == 0:
                    continue

                left_entropy = entropy(y[left_idx])
                right_entropy = entropy(y[right_idx])

                #weighted average of child entropies
                child_entropy = (
                    len(y[left_idx]) / len(y) * left_entropy +
                    len(y[right_idx]) / len(y) * right_entropy
                )

                # information gain = how much entropy is reduced
                gain = parent_entropy - child_entropy

                #keep the best split
                if gain > best_gain:
                    best_gain = gain
                    best_feature = f
                    best_threshold = thr

        return best_feature, best_threshold

    # the class that appears the most in y
    def _most_common(self, y):
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    #if leaf node then return its class
    def predict_one(self, x, node):
        if node.value is not None:
            return node.value

        #otherwise go left or right based on the split
        if x[node.feature] <= node.threshold:
            return self.predict_one(x, node.left)
        else:
            return self.predict_one(x, node.right)

    #finally predict each row in X
    def predict(self, X):
        return np.array([self.predict_one(x, self.root) for x in X])




# Load the data from CSVs

# Use the paths we defined earlier
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)



# split into features and labels

# all columns named feature_0, feature_1, ..., feature_N
feature_cols = [c for c in train_df.columns if c.startswith("feature_")]

# Training features and labels
X = train_df[feature_cols].values.astype(np.float32)
y = train_df["label"].values.astype(int)

# Test features and ids
X_test = test_df[feature_cols].values.astype(np.float32)
test_ids = test_df["id"].values

print("X shape:", X.shape)        
print("y shape:", y.shape)        
print("X_test shape:", X_test.shape) 


# Create and train the tree
tree = DecisionTree(
    max_depth=12,          # you can play with this
    min_samples_split=200  # and this
)

# Builds the tree
tree.fit(X, y)

print("Training finished!")



# Predict on test data
preds = tree.predict(X_test)

print("Predictions shape:", preds.shape)
print("First 10 predictions:", preds[:10])


# Create submission.csv (for final results)
submission = pd.DataFrame({
    "id": test_ids,
    "label": preds
})

submission.to_csv("submission_ensemble_tree.csv", index=False)
print("Saved submission.csv")
print(submission.head())
