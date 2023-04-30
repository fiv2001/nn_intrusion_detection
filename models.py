import abc
import lightgbm as lgb

class BaseModel(abc.ABC):
    @abc.abstractmethod
    def fit(X_train, y_train):
        pass

    @abc.abstractmethod
    def predict(X_test):
        pass


class LGBMModel(BaseModel):
    def __init__(self, num_leaves, n_estimators):
        self.model = lgb.LGBMClassifier(num_leaves=num_leaves, n_estimators=n_estimators)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)



