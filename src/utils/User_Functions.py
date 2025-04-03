import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import classification_report
from sklearn.metrics import silhouette_samples, silhouette_score
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns
from scipy.stats import boxcox

# User Functions

def try_GPU():
    """
    Checks if a GPU is available and sets it as the visible device for TensorFlow if found.
    If no GPU is available, it defaults to using the CPU.

    This function lists the physical devices available for TensorFlow and selects the first GPU
    if one is available. If a GPU is not found, it will print a message indicating that the CPU is being used.

    Prints:
        - "Using GPU" if a GPU is available and set for use.
        - "Using CPU" if no GPU is available, and the CPU is being used.
    """
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.set_visible_devices(physical_devices[0], 'GPU')
        print("Using GPU")
    else:
        print("Using CPU.")


def type_features(df):
    """
    Classifies the columns of a DataFrame into three categories: 
    numerical features, categorical features, and ID features.

    Parameters:
    df (pd.DataFrame): The DataFrame to classify.

    Returns:
    list: Three lists of column names:
          - features_num: List of numerical column names.
          - features_cat: List of categorical column names.
          - features_id: List of columns considered as IDs (categorical columns with numeric values).
    """
    
    # Initializing lists for each type of feature
    features_num = []  # For numerical columns
    features_cat = []  # For categorical columns
    features_id = []   # For categorical columns that could be IDs (numbers representing IDs)

    # Loop to classify the columns
    for column in df.columns:
        dtype = df[column].dtype
        
        # Ignore datetime columns
        if pd.api.types.is_datetime64_any_dtype(dtype):
            continue
        
        # Check if the column is numerical
        if pd.api.types.is_numeric_dtype(dtype):
            features_num.append(column)
        # Check if the column is categorical (object or categories)
        elif pd.api.types.is_object_dtype(dtype):
            # Check if the column can be considered an ID (usually numbers represented as strings)
            if df[column].str.isnumeric().all():
                features_id.append(column)
            else:
                features_cat.append(column)

    # Return the lists instead of a dictionary
    return features_num, features_cat, features_id


# Function to plot all histograms in a single figure
def plot_histograms(df, features_num=None):
    """
    Plots histograms with Kernel Density Estimation (KDE) for each numeric feature in the given DataFrame.
    
    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the data.
    features_num (list, optional): List of numeric feature columns to plot. 
                                     If None, all numeric columns from the DataFrame will be used.
    
    This function creates subplots for each numeric feature and displays them in a grid layout,
    adjusting the number of rows to fit the number of features.
    """
    # If features_num is not passed, get all numeric columns
    if features_num is None:
        features_num = df.select_dtypes(include=['number']).columns.tolist()

    num_cols = len(features_num)
    rows = math.ceil(num_cols / 3)  # Adjusts to up to 3 plots per row

    fig, axes = plt.subplots(rows, 3, figsize=(15, 5 * rows))  # Adjustable size
    axes = axes.flatten()  # Flattens the matrix into an array for iteration

    for i, col in enumerate(features_num):
        sns.histplot(df[col], kde=True, bins=30, ax=axes[i])
        axes[i].set_title(f'Histogram and KDE of {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')

    # Hide empty axes if any
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()  # Adjust layout spacing
    plt.show()

def target_encoding_kfold(train_set, test_set, categorical_features, target, n_splits=5, smoothing=0.3, seed=42):
    """
    Applies Target Encoding with K-Fold to prevent data leakage.
    
    Parameters:
    - train_set: Training DataFrame
    - test_set: Testing DataFrame
    - categorical_features: List of categorical columns to be encoded
    - target: Name of the target variable
    - n_splits: Number of splits for cross-validation
    - smoothing: Smoothing factor to avoid overfitting
    - seed: Random seed for reproducibility
    
    Returns:
    - Encoded train_set (without original categorical columns)
    - Encoded test_set (without original categorical columns)
    - List of encoded column names
    """
    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    # Copy the datasets
    train_encoded = train_set.copy()
    test_encoded = test_set.copy()
    
    # List to store new target encoding column names
    encoded_columns = []
    
    # Global mean of the target variable
    global_mean = train_set[target].mean()
    
    for col in categorical_features:
        category_means = {}
        train_encoded[col + "_te"] = 0.0
        test_encoded[col + "_te"] = 0.0
        encoded_columns.append(col + "_te")
        
        for train_idx, valid_idx in skf.split(train_set, train_set[target]):
            train_fold = train_set.iloc[train_idx]
            valid_fold = train_set.iloc[valid_idx].copy()
            
            category_stats = train_fold.groupby(col, observed=False)[target].agg(['mean', 'count'])
            category_means_smooth = (category_stats['mean'] * category_stats['count'] + global_mean * smoothing) / (category_stats['count'] + smoothing)
            category_means.update(category_means_smooth.to_dict())
            
            train_encoded.loc[train_set.index[valid_idx], col + "_te"] = valid_fold[col].map(category_means).fillna(global_mean).values
        
        test_encoded[col + "_te"] = test_set[col].map(category_means).fillna(global_mean).values
        
        # Remove the original categorical column
        train_encoded.drop(columns=[col], inplace=True)
        test_encoded.drop(columns=[col], inplace=True)
    
    return train_encoded, test_encoded, encoded_columns

def target_encoding_kfold_regression(train_set, test_set, categorical_features, target, n_splits=5, smoothing=0.3, seed=42):
    """
    Applies Target Encoding with K-Fold for regression to prevent data leakage.
    
    Parameters:
    - train_set: Training DataFrame
    - test_set: Testing DataFrame
    - categorical_features: List of categorical columns to be encoded
    - target: Name of the target variable (continuous)
    - n_splits: Number of splits for cross-validation
    - smoothing: Smoothing factor to avoid overfitting
    - seed: Random seed for reproducibility
    
    Returns:
    - Encoded train_set
    - Encoded test_set
    """
    # Use KFold instead of StratifiedKFold for regression
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    # Initialize the DataFrame to store the encoded results
    train_encoded = train_set.copy()
    test_encoded = test_set.copy()
    
    # Initialize a list to collect all the new target encoding columns
    encoded_columns = []
    
    # Calculate the global mean of the target
    global_mean = train_set[target].mean()
    
    for col in categorical_features:
        # Create a dictionary to store the category means
        category_means = {}
        
        # Create the new column for target encoding
        train_encoded[col + "_te"] = 0.0
        test_encoded[col + "_te"] = 0.0
        
        # Add the new column to the list of encoded columns
        encoded_columns.append(col + "_te")
        
        # Loop through the folds
        for train_idx, valid_idx in kf.split(train_set):
            # Split into train and validation
            train_fold = train_set.iloc[train_idx]
            valid_fold = train_set.iloc[valid_idx].copy()  
            valid_fold.reset_index(drop=True, inplace=True)

            # Calculate the mean and count of the target for each category
            category_stats = train_fold.groupby(col, observed=False)[target].agg(['mean', 'count'])
            category_means_smooth = (category_stats['mean'] * category_stats['count'] + global_mean * smoothing) / (category_stats['count'] + smoothing)
            
            # Update the dictionary with the smoothed means
            category_means.update(category_means_smooth.to_dict())
            
            # Apply target encoding to the validation fold using .map(), ensuring proper indexing
            train_encoded.reset_index(drop=True, inplace=True)
            train_encoded.loc[valid_idx, col + "_te"] = valid_fold[col].map(category_means).fillna(global_mean).values
        
        # Apply target encoding to the test set, making sure missing categories are handled with global_mean
        test_encoded[col + "_te"] = test_set[col].map(category_means).fillna(global_mean).values
        
    return train_encoded, test_encoded, encoded_columns

def suggest_best_transformations(df, features_num, skew_threshold=0.5):
    """
    Suggests the best transformation for each numeric feature in a DataFrame based on skewness.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the data.
    features_num (list): List of numeric feature columns to evaluate for skewness and transformations.
    skew_threshold (float, optional): The threshold below which no transformation is suggested. Default is 0.5.

    Returns:
    dict: A dictionary with column names as keys and the suggested transformation as values. If no transformation
          is suggested, the value will be None.

    The function evaluates the skewness of each numeric feature and suggests transformations if the skewness 
    is above the defined threshold. It tests the following transformations: square root, cube root, log1p, and Box-Cox.
    The transformation that results in the least skewness (closest to 0) is suggested.
    """
    transformation_suggestions = {}

    print("Suggested transformations:\n")
    for col in features_num:
        data = df[col].dropna()  # Remove NaN values
        original_skew = data.skew()
        print(f"Column '{col}': Original skew: {original_skew:.3f}")

        if abs(original_skew) < skew_threshold:
            print(f"  No transformation needed (below threshold)\n")
            continue

        transformations = {
            "sqrt": lambda x: np.sqrt(x) if np.all(x >= 0) else None,
            "cbrt": lambda x: np.cbrt(x),
            "log1p": lambda x: np.log1p(x) if np.all(x >= 0) else None,
            "boxcox": lambda x: boxcox(x + 1)[0] if np.all(x > 0) else None  # Box-Cox exige valores positivos
        }

        best_transform = None
        best_skew = abs(original_skew)
        tested_transforms = []

        for name, func in transformations.items():
            try:
                transformed = func(data)
                if transformed is not None:
                    new_skew = pd.Series(transformed).skew()
                    tested_transforms.append((name, new_skew))
                    print(f"  {name} -> new skew {new_skew:.3f}")

                    if abs(new_skew) < best_skew:  # Se melhorar a skew, armazena a melhor transformação
                        best_skew = abs(new_skew)
                        best_transform = name
            except Exception as e:
                print(f"  {name} failed -> {e}")
                continue

        if best_transform:
            print(f"  Column '{col}': {best_transform} transformation suggested (new skew: {best_skew:.3f})\n")
            transformation_suggestions[col] = best_transform
        else:
            print(f"  No transformation suggested (all skews worsened)\n")

    return transformation_suggestions


def apply_transformations(train_encoded: pd.DataFrame, test_encoded: pd.DataFrame, transformations: dict):
    """
    Applies specified transformations to the given columns in train and test datasets.

    Parameters:
    train_encoded (pd.DataFrame): The training dataset.
    test_encoded (pd.DataFrame): The testing dataset.
    transformations (dict): A dictionary mapping column names to transformations.

    Returns:
    tuple: Transformed train_encoded and test_encoded DataFrames.
    """
    for col, transform in transformations.items():
        if col in train_encoded and col in test_encoded:
            if transform == "sqrt":
                train_encoded[col] = np.sqrt(train_encoded[col])
                test_encoded[col] = np.sqrt(test_encoded[col])
            elif transform == "cbrt":
                train_encoded[col] = np.cbrt(train_encoded[col])
                test_encoded[col] = np.cbrt(test_encoded[col])
            elif transform == "log1p":
                train_encoded[col] = np.log1p(train_encoded[col])
                test_encoded[col] = np.log1p(test_encoded[col])
            elif transform == "boxcox":
                if (train_encoded[col] > 0).all() and (test_encoded[col] > 0).all():
                    train_encoded[col], lambda_train = boxcox(train_encoded[col] + 1)
                    test_encoded[col] = boxcox(test_encoded[col] + 1, lmbda=lambda_train)
                else:
                    print(f"Skipping Box-Cox for '{col}' due to non-positive values.")
            else:
                print(f"Unknown transformation '{transform}' for column '{col}'.")
    
    return train_encoded, test_encoded

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Train a given model on the training data and evaluate its performance on both 
    the training and test datasets by printing the classification reports.

    Args:
        model (sklearn.base.BaseEstimator): The machine learning model to be evaluated.
        X_train (pandas.DataFrame or numpy.ndarray): The features of the training data.
        y_train (pandas.Series or numpy.ndarray): The target values of the training data.
        X_test (pandas.DataFrame or numpy.ndarray): The features of the test data.
        y_test (pandas.Series or numpy.ndarray): The target values of the test data.
    
    Prints:
        Classification report for both training and test sets including precision, recall, 
        f1-score, and support for each class.
    """
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Print the classification report for train and test data
    print(f"Classification Report (Train) for {model.__class__.__name__}:")
    print(classification_report(y_train, y_pred_train))
    
    print(f"Classification Report (Test) for {model.__class__.__name__}:")
    print(classification_report(y_test, y_pred_test))


def scale_features(X_train, X_test):
    """
    Scale the features of the training and test datasets using StandardScaler, 
    ensuring that the original index of the data is preserved.

    Args:
        X_train (pandas.DataFrame or numpy.ndarray): The features of the training data.
        X_test (pandas.DataFrame or numpy.ndarray): The features of the test data.
    
    Returns:
        tuple: A tuple containing the scaled training features and test features as DataFrames.
    
    The function uses StandardScaler to standardize the features, ensuring each feature 
    has zero mean and unit variance.
    """
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    
    scaler = StandardScaler()
    
    # Keeping original index
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    
    return X_train_scaled, X_test_scaled



def plot_silhouette(X, labels):

    """
    Plots a Silhouette Score chart for clusters.

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Data points to be clustered.
    labels : array-like, shape (n_samples,)
        Cluster labels, with -1 indicating outliers.

    Returns:
    --------
    None
        Displays the silhouette score plot.

    Notes:
    ------
    Visualizes silhouette values for each cluster, with a red line showing the average score.
    """
    silhouette_vals = silhouette_samples(X, labels)
    silhouette_avg = silhouette_score(X, labels)

    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != -1]  # Remove outliers
    y_lower, y_upper = 0, 0

    plt.figure(figsize=(10, 6))
    
    for i, label in enumerate(unique_labels):
        cluster_silhouette_vals = silhouette_vals[labels == label]
        cluster_silhouette_vals.sort()

        y_upper += len(cluster_silhouette_vals)
        plt.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_vals, alpha=0.7)

        plt.text(-0.05, y_lower + 0.5 * len(cluster_silhouette_vals), str(label))
        y_lower = y_upper

    plt.axvline(x=silhouette_avg, color="red", linestyle="--", label="Average Silhouette")
    plt.title("Silhouette Score for HDBSCAN Clusters", fontsize=14)
    plt.xlabel("Silhouette Coefficient", fontsize=12)
    plt.ylabel("Cluster", fontsize=12)
    plt.legend()
    plt.show()

def plot_feature_importance_logistic(best_model, feature_names, model_name):
    """
    Plots the absolute values of the coefficients as feature importances for Logistic Regression.
    
    Parameters:
    best_model : Trained Logistic Regression model.
    feature_names : List of feature names corresponding to model input.
    model_name : String representing the model name for title purposes.
    """
    importances = np.abs(best_model.coef_).flatten()
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title(f"{model_name} - Feature Importances")
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.show()

def plot_feature_importance(best_model, feature_names, model_name):

    """
    Plots the feature importances of a trained model.
    
    Parameters:
    best_model : Trained model object with feature_importances_ attribute.
    feature_names : List of feature names corresponding to model input.
    model_name : String representing the model name for title purposes.
    """
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title(f"{model_name} - Feature Importances")
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.show()