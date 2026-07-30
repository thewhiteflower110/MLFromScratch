import numpy as np

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature        # Index of feature to split on
        self.threshold = threshold    # Threshold value for feature
        self.left = left              # Left child (Node)
        self.right = right            # Right child (Node)
        self.value = value            # Class label (if leaf node)
    
    def is_leaf_node(self):
        return self.value is not None

def gini_impurity(y):
    """Calculate Gini Impurity for a label array y."""
    if len(y) == 0:
        return 0.0
    
    p = np.bincount(y) / len(y)
    return 1.0 - np.sum(p ** 2)

def _information_gain( y, X_column, threshold, current_impurity):
    left_idxs = X_column <= threshold
    right_idxs = ~left_idxs

    if len(y[left_idxs]) == 0 or len(y[right_idxs]) == 0:
        return 0

    # Weighted average of child impurities
    n = len(y)
    n_l, n_r = len(y[left_idxs]), len(y[right_idxs])
    gini_l = gini_impurity(y[left_idxs])
    gini_r = gini_impurity(y[right_idxs])
    child_impurity = (n_l / n) * gini_l + (n_r / n) * gini_r

    # Information gain is impurity reduction
    ig = current_impurity - child_impurity
    return ig

class DecisionTreeClassifier:
    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.root = None
    
    def fit(self, X, y):
        """Build the decision tree from training set (X, y)."""
        self.root = self._build_tree(X, y)
    
    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # 1. Check stopping criteria
        if (depth >= self.max_depth or 
            n_labels == 1 or 
            n_samples < self.min_samples_split):
            # Return a leaf node containing the most common class
            most_common_label = np.bincount(y).argmax()
            return Node(value=most_common_label)

        # 2. Find the best split
        best_feature, best_thresh = self._best_split(X, y, n_features)

        # If no valid split improved impurity, create a leaf
        if best_feature is None:
            most_common_label = np.bincount(y).argmax()
            return Node(value=most_common_label)
        
        # 3. Create child nodes recursively
        left_idxs = X[:, best_feature] <= best_thresh
        right_idxs = ~left_idxs

        left_child = self._build_tree(X[left_idxs], y[left_idxs], depth + 1)
        right_child = self._build_tree(X[right_idxs], y[right_idxs], depth + 1)

        return Node(feature=best_feature, threshold=best_thresh, left=left_child, right=right_child)

    def _best_split(self, X, y, n_features):
        best_gain = -1.0
        split_idx, split_threshold = None, None

        current_impurity = gini_impurity(y)
        for feature_idx in range(n_features):
            X_column = X[:, feature_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = _information_gain(y, X_column, threshold, current_impurity)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature_idx
                    split_threshold = threshold
        return split_idx, split_threshold

    def predict(self,X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

