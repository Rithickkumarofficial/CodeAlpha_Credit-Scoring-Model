
from sklearn.ensemble import RandomForestClassifier

def train_random_forest(X_train, y_train):
    """
    Trains Random Forest model for credit scoring
    """
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model
