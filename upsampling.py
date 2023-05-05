import abc
import imblearn.over_sampling as ups

class BaseUpsampling(abc.ABC):
    @abc.abstractmethod
    def fit_resample(self, X, y):
        pass


class Smote(BaseUpsampling):
    def __init__(self, sampling_strategy):
        self.upsampling = ups.SMOTE(sampling_strategy=sampling_strategy)

    def fit_resample(self, X, y):
        return self.upsampling.fit_resample(X, y)

