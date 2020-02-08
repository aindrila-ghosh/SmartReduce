import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
RANDOM_STATE = 42

def train_ML_models(data_target, points):

    """Trains KNeighborsClassifier with the embedding and computes the prediction accuracy.
    
    Parameters
    ----------
    data_target : list
            Original labels
    points : nD array 
            embedding
    Returns
    -------
    accuracy_knn : float 
            value of KNN accuracy
    """
    
    train_features, test_features, train_labels, test_labels = train_test_split(points, data_target, test_size = 0.25, random_state = RANDOM_STATE)

    knn = KNeighborsClassifier(3)
    knn.fit(train_features, train_labels.ravel())
    predictions_knn = knn.predict(test_features)
    accuracy_knn = knn.score(test_features, test_labels)

    return accuracy_knn


def run_k_fold_cross_validation(train_features, train_labels, test_features, test_labels):

    """Runs k_fold_cross_validation on the dataset.
    
   Parameters
    ----------
    train_features : nD array 
                    Original features
    train_labels : nD array 
                    Original labels
    test_features : nD array 
                    Original test features
    test_labels : nD array 
                    Original test labels
    Returns
    -------
    accuracy_xgb : float 
                value of XGBoost accuracy
    """

    xgbc = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
    xgbc.fit(train_features, train_labels)
    predictions_xgb = xgbc.predict(test_features)
    accuracy_xgb = xgbc.score(test_features, test_labels)

    return accuracy_xgb
