
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

# --- 1. LOADING THE DATA ---
# Adjust the paths to "UCI HAR Dataset"
PATH = "UCI HAR Dataset/"  # Replace with your actual path
features_path = PATH + "features.txt"
activity_labels_path = PATH + "activity_labels.txt"
X_train_path = PATH + "train/X_train.txt"
y_train_path = PATH + "train/y_train.txt"
X_test_path = PATH + "test/X_test.txt"
y_test_path = PATH + "test/y_test.txt"

# Load feature names, This appends the column index to any duplicate names.
features_df = pd.read_csv(features_path, sep="\s+", header=None, names=["idx",
"feature"])
feature_names = features_df["feature"].tolist()

# his appends the column index to any duplicate names.
features_df["feature"] = features_df["feature"].astype(str) + "_" + features_df.index.astype(str)
feature_names = features_df["feature"].tolist()

# Load activity labels (mapping IDs 1-6 to string names)
activity_labels_df = pd.read_csv(activity_labels_path, sep="\s+", header=None, names=["id", "activity"])
activity_map = dict(zip(activity_labels_df["id"], activity_labels_df["activity"]))

# Load train/test sets
X_train = pd.read_csv(X_train_path, sep="\s+", header=None, names=feature_names)
y_train = pd.read_csv(y_train_path, sep="\s+", header=None, names=["Activity"])
X_test = pd.read_csv(X_test_path, sep="\s+", header=None, names=feature_names)
y_test = pd.read_csv(y_test_path, sep="\s+", header=None, names=["Activity"])

# Map the activity IDs to their names
y_train["Activity"] = y_train["Activity"].map(activity_map)
y_test["Activity"] = y_test["Activity"].map(activity_map)

# --- 2. CONVERT MULTI-CLASS TO BINARY ---
def to_binary_label(activity):
  if activity in ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS"]:
    return 1 # Active
  else:
    return 0 # Inactive

y_train["Binary"] = y_train["Activity"].apply(to_binary_label)
y_test["Binary"] = y_test["Activity"].apply(to_binary_label)

# Load subject IDs from files
subject_train_path = PATH + "train/subject_train.txt"
subject_test_path = PATH + "test/subject_test.txt"

subject_train = pd.read_csv(subject_train_path, header=None, names=['Subject'])
subject_test = pd.read_csv(subject_test_path, header=None, names=['Subject'])

print("Subject ids loaded")

# Recombine X_train, X_test, y_train, and y_test
X = pd.concat([X_train,X_test]) # combines to a new variable
y = pd.concat([y_train,y_test]) # combines to a new variable
subject = pd.concat([subject_train, subject_test])

# 3: Make a new train and test split that does not have overlap with subject test
#   *   Function to filter data by subject

# Function to filter data by subject
def filter_by_subject(X, y, subject, subject_list): #this takes 4 arguments.
    combined_data = pd.concat([X, y, subject], axis=1) # Combines/adds X and Y and subject

    # Filters X and y to only subjects that dont overlap between test or train
    filtered_data = combined_data[combined_data['Subject'].isin(subject_list)] #filters the activity values in list.

    X_filtered = filtered_data.drop(['Activity', 'Subject'], axis=1)  # Remove activity as feature
    y_filtered = filtered_data['Activity'] # Get activity column as target

    return X_filtered, y_filtered

# from sklearn.model_selection import train_test_split
TEST_SIZE = .2 # test size percent
RANDOM = 42 # random seed

subjects = np.unique(subject)
train_subjects, test_subjects = train_test_split(subjects, test_size=TEST_SIZE, random_state=RANDOM)

X_train, y_train = filter_by_subject(X, y, subject, train_subjects)
X_test, y_test = filter_by_subject(X, y, subject, test_subjects)

y_train["Binary"] = y_train.apply(to_binary_label)
y_test["Binary"] = y_test.apply(to_binary_label)

print ("after combining train split shapes", X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Scale again
# --- 3. FEATURE SCALING ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#PCA Reduct again
pca = PCA(n_components=300, random_state=42) #This is setting the PCA to 50 components
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# --- 5. BASELINE SVM MODEL TRAINING ---
print("\n--- Training Baseline SVM Models ---")
kernels = ['linear', 'poly', 'rbf']
baseline_results = {}

for kernel in kernels:
    print(f"\nTraining SVM with kernel: {kernel}")
    svm_model = SVC(kernel=kernel, random_state=42)  # Set random_state for reproducibility
    svm_model.fit(X_train_pca, y_train["Binary"].values)
    y_pred = svm_model.predict(X_test_pca)

    accuracy = accuracy_score(y_test["Binary"], y_pred)
    precision = precision_score(y_test["Binary"], y_pred)
    recall = recall_score(y_test["Binary"], y_pred)
    f1 = f1_score(y_test["Binary"], y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    baseline_results[kernel] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# --- 6. HYPERPARAMETER TUNING WITH GRIDSEARCHCV ---
print("\n--- Performing Hyperparameter Tuning with GridSearchCV ---")

# Define the parameter grid
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [0.001, 0.1, 1],
    'kernel': ['rbf', 'poly', 'linear']
}

# Create GridSearchCV object
grid_search = GridSearchCV(SVC(), param_grid, cv=3, scoring='accuracy', verbose=1, n_jobs=-1) #n_jobs=-1 means use all processors

# Fit the model
grid_search.fit(X_train_pca, y_train["Binary"])

# Print the best parameters
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

# --- 7. EVALUATE BEST MODEL ---
print("\n--- Evaluating the Best Model ---")
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test_pca)

accuracy_best = accuracy_score(y_test["Binary"], y_pred_best)
precision_best = precision_score(y_test["Binary"], y_pred_best)
recall_best = recall_score(y_test["Binary"], y_pred_best)
f1_best = f1_score(y_test["Binary"], y_pred_best)
confusion = confusion_matrix(y_test["Binary"], y_pred_best)

print(f"Accuracy: {accuracy_best:.4f}")
print(f"Precision: {precision_best:.4f}")
print(f"Recall: {recall_best:.4f}")
print(f"F1-score: {f1_best:.4f}")
print("\nConfusion Matrix:\n", confusion)
