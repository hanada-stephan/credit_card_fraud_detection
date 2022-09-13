from sklearn.metrics import accuracy_score,\
                            precision_score,\
                            recall_score
                            

def validation_scores(y_test, y_pred):

    """Print accuracy, precision and recall scores for classification models
    
    Args:
        y_test (Pandas series, nparray): Target of test data set
        y_pred (Pandas series, nparray): Target predicted

    Returns:
        Three prints for the model scores
    """ 
    print("Accuracy score: ", accuracy_score(y_test, y_pred))
    print("Precision score: ",precision_score(y_test, y_pred))
    print("Recall score: ", recall_score(y_test, y_pred))