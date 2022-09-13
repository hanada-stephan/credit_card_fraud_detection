from sklearn.model_selection import StratifiedShuffleSplit

def run_model(model, X_train, X_test, y_train):
    """Run sklearn models
    
    Args:
        model (sklearn model): model to run
        X_train (Pandas series, nparray): Training data set
        X_test (Pandas series, nparray): Test data set
        y_train (Pandas series, nparray): Target of training data set

    Returns:
        array with predictions
    """ 
    model_inst = model.fit(X_train, y_train)
    y_pred = model_inst.predict(X_test)
    return y_pred


def validator_stratified_shuffle_split(X, y):
    """Do the stratified shuffle split for unbalanced data set
    
    Args:
        X (Pandas series, nparray): Independent variables 
        y (Pandas series, nparray): Target of the data set

    Returns:
        Independent and dependent variables for train and test set
    """ 
    validator = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
    for train_id, test_id in validator.split(X, y):
        X_train, X_test = X[train_id], X[test_id]
        y_train, y_test = y[train_id], y[test_id]
    return X_train, X_test, y_train, y_test
    