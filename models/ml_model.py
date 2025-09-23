import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import logging

logger = logging.getLogger(__name__)

class MLPredictor:
    def __init__(self):
        self.model = self._train_model()
    
    def _train_model(self):
        # Dummy training data (same)
        data = pd.DataFrame({
            'moisture': np.random.uniform(20, 80, 100),
            'temp': np.random.uniform(20, 40, 100),
            'crop': np.random.choice(['rice', 'wheat'], 100),
            'pest_risk': np.random.choice(['low', 'medium', 'high'], 100)
        })
        data['crop_rice'] = (data['crop'] == 'rice').astype(int)
        X = data[['moisture', 'temp', 'crop_rice']]
        y = pd.get_dummies(data['pest_risk']).idxmax(axis=1)  # Numeric labels
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = RandomForestClassifier(n_estimators=10)
        model.fit(X_train, y_train)
        joblib.dump(model, 'models/pest_model.pkl')
        logger.debug("[DEV] ML model trained with dummy data.")
        return model
    
    def predict_pest_risk(self, moisture, temp, crop):
        crop_rice = 1 if crop == 'rice' else 0
        features = np.array([[moisture, temp, crop_rice]])
        pred = self.model.predict(features)[0]
        risks = {0: 'low', 1: 'medium', 2: 'high'}
        risk = risks.get(pred, 'low')
        logger.debug(f"[DEV] Pest prediction: moisture={moisture}, temp={temp}, crop={crop} -> {risk}")
        return risk