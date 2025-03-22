import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

def train_cost_sensitive_churn_model(csv_filepath, intervention_cost=10, random_state=42):
    """
    Loads the data, preprocesses it, and trains a cost-sensitive decision tree classifier.
    
    Parameters:
        csv_filepath (str): Path to the CSV file containing the dataset.
        intervention_cost (float): Fixed cost for customer intervention.
        random_state (int): Random state for reproducibility.
    
    Returns:
        clf (DecisionTreeClassifier): Trained decision tree classifier.
        features (list): List of feature names used in the model.
    """
    # Load and preprocess the dataset
    data = pd.read_csv(csv_filepath)

    # Encode categorical features
    data_encoded = pd.get_dummies(
        data, 
        columns=['Gender', 'Promotion_Response', 'Email_Opt_In'], 
        drop_first=True
    )

    # Feature Engineering
    data_encoded['Return_Ratio'] = data_encoded['Num_of_Returns'] / data_encoded['Num_of_Purchases'].replace(0, 1)
    data_encoded['Purchase_Frequency'] = data_encoded['Num_of_Purchases'] / data_encoded['Last_Purchase_Days_Ago'].replace(0, 1)
    data_encoded['CLV'] = (data_encoded['Average_Transaction_Amount'] *
                           data_encoded['Purchase_Frequency'] *
                           data_encoded['Years_as_Customer']).fillna(0)

    # Intervention cost
    data_encoded['Intervention_Cost'] = intervention_cost

    # Utility values calculation
    data_encoded['Utility_True_Positive'] = data_encoded['CLV'] - intervention_cost
    data_encoded['Utility_False_Positive'] = - intervention_cost
    data_encoded['Utility_False_Negative'] = - data_encoded['CLV']
    data_encoded['Utility_True_Negative'] = 0

    # Identify features dynamically
    encoded_columns = [col for col in data_encoded.columns if 
                       col.startswith('Gender_') or
                       col.startswith('Promotion_Response_') or
                       col.startswith('Email_Opt_In_')]

    features = ['Return_Ratio', 'Purchase_Frequency', 'CLV'] + encoded_columns

    # Split dataset
    X = data_encoded[features]
    y = data_encoded['Target_Churn'].astype(int)

    X_train, _, y_train, _, util_train, _ = train_test_split(
        X, y,
        data_encoded[['Utility_True_Positive', 'Utility_False_Positive', 
                      'Utility_False_Negative', 'Utility_True_Negative']],
        test_size=0.3, random_state=random_state
    )

    # Sample weights for cost-sensitive learning
    sample_weights = np.where(
        y_train == 1,
        util_train['Utility_True_Positive'].abs(),
        util_train['Utility_False_Negative'].abs()
    )

    # Train the Decision Tree Classifier
    clf = DecisionTreeClassifier(random_state=random_state)
    clf.fit(X_train, y_train, sample_weight=sample_weights)

    # Return the classifier and features list
    return clf, features
