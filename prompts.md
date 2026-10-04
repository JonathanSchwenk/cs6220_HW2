# Prompts used

## Linear regression
I am building a linear regression model completely from scratch using numpy, avoiding any external regression libraries. For the first step of the training process, I need a helper function to prepare the data.
Please write a Python function that takes a pandas DataFrame where the columns represent the feature \(x\), any additional transformed features (like \(f_1(x)\), \(f_2(x)\), etc.), and a final column named 'y' for the target variable.
The function should output two things:
1. A matrix \(X\) (as a numpy array) that includes all the feature columns plus an added leading column of 1s to account for the intercept.
2. A vector \(Y\) (as a numpy array) extracted from the 'y' column.
Please keep the scope strictly to this initial data conversion step. Do not compute the weight vector \(\beta \) or write any further parts of the regression algorithm yet.

I am ready for the second step of building my linear regression model from scratch using numpy.
I need a function that takes the matrix \(X\) and the vector \(Y\) produced by my build_xy helper function and computes the weight vector \(\beta \).
Please implement the normal equation formula: \(\beta = (X^T X)^{-1} X^T Y\) using numpy's linear algebra tools. Do not use any external regression libraries.
Keep the scope strictly to this calculation step inside the training process. Do not write the predict function or any other parts of the model yet.

I am ready for the next phase of the linear regression model: implementing the prediction step.
I need a function that takes two inputs: the weight vector \(\beta \) computed during training, and a test pandas DataFrame formatted exactly like the training data (containing the feature columns, but without the target column 'y').
The function should build the test feature matrix \(X\) in the exact same manner as before—including the leading column of 1s for the intercept—and then compute the predictions using the matrix multiplication \(\hat{y} = X \cdot \beta\). The output should be a numpy array containing one prediction per row of the input data.
Please write this using numpy only, without using any external regression or evaluation libraries, as I am not ready to evaluate the model yet.

I am ready for the final piece of the linear regression model: evaluating the predictions.
I need an evaluation function that takes two inputs: a numpy array of model predictions (\(\^{y}\)) and a numpy array of the true y values. The function should calculate and return a single number representing the Root Mean Squared Error (RMSE).
Please implement this by following these specific steps:
1. Calculate the error for each prediction and square it.
2. Find the mean of those squared errors.
3. Take the square root of that mean.
Please write this function using numpy only, without importing any external evaluation libraries like sklearn.metrics.

## Regression tree
Using only NumPy, write a function that finds the best split for a node in a regression tree.

The input should contain the records currently in the node, including one or more feature columns and the target values y.

For each feature, sort its unique values and consider candidate split thresholds at the midpoint between consecutive values. For each threshold, divide the records into a left group where the feature value is less than the threshold and a right group where it is greater than or equal to the threshold.

Score each split by calculating the mean y value for each group and then computing the sum of squared errors from those means for both groups. Add the left and right squared errors together. The split with the smallest total error should be selected.

Return the feature used for the best split, the threshold, and the resulting total squared error. Ignore candidate splits that would leave either side empty.

Use NumPy only. Do not use sklearn decision trees or other existing tree implementations. Do not implement the full tree or fit method yet.

Using the `best_split` function from the previous step, create the node structure and `RegressionTree` class with its `fit(dataset, min_records)` method.

Create a `Node` class that can represent either a leaf or an internal node. A leaf should store its prediction, which is the mean y value of the records in that node. An internal node should store the feature index and threshold used for its split, along with its left and right child nodes. Also store the number of records that reached each node so I can use it later to examine leaf sizes.

For `fit`, the input dataset is a DataFrame where the last column is y and the other columns are the features. Convert the dataset into a NumPy feature array X and target vector y. Do not include the extra column of 1s that was used for linear regression.

Build the tree recursively. For each node, if the number of records is less than `min_records`, create a leaf whose prediction is the mean y value of those records. Otherwise, call the existing `best_split` function.

If `best_split` does not find a valid split, create a leaf containing the mean y. If a valid split is found, divide the records using the same rule as `best_split`: feature values less than the threshold go to the left child, and values greater than or equal to the threshold go to the right child. Recursively repeat this process for both children.

Use the `Node` class and `best_split` function rather than rewriting the split logic inside `fit`.

Use NumPy for the tree calculations and do not use sklearn or any existing decision-tree library. Only implement the node structure, `RegressionTree` class, and `fit` method in this step. Do not implement `predict` or `visualize` yet.
