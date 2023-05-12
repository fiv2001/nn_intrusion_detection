import abc
import imblearn.over_sampling as ups

from config import CONFIG

from sklearn.cluster import MiniBatchKMeans

class BaseUpsampling(abc.ABC):
    @abc.abstractmethod
    def fit_resample(self, X, y):
        pass


class NoneUpsampling(BaseUpsampling):
    def fit_resample(self, X, y):
        return X, y


class Random(BaseUpsampling):
    def __init__(self, sampling_strategy):
        self.upsampling = ups.RandomOverSampler(sampling_strategy=sampling_strategy)

    def fit_resample(self, X, y):
        return self.upsampling.fit_resample(X, y)


class Smote(BaseUpsampling):
    def __init__(self, sampling_strategy):
        self.upsampling = ups.SMOTE(sampling_strategy=sampling_strategy)

    def fit_resample(self, X, y):
        return self.upsampling.fit_resample(X, y)


class BorderlineSmote(BaseUpsampling):
    def __init__(self, sampling_strategy):
        self.upsampling = ups.BorderlineSMOTE(sampling_strategy=sampling_strategy)

    def fit_resample(self, X, y):
        return self.upsampling.fit_resample(X, y)


class SvmSmote(BaseUpsampling):
    def __init__(self, sampling_strategy):
        self.upsampling = ups.SVMSMOTE(sampling_strategy=sampling_strategy)

    def fit_resample(self, X, y):
        return self.upsampling.fit_resample(X, y)


class KMeansSmote(BaseUpsampling):
    def __init__(self, sampling_strategy, cluster_balance_threshold, n_init):
        self.upsampling = ups.KMeansSMOTE(
            sampling_strategy=sampling_strategy,
            cluster_balance_threshold=cluster_balance_threshold,
            kmeans_estimator=MiniBatchKMeans(n_init=n_init))

    def fit_resample(self, X, y):
        return self.upsampling.fit_resample(X, y)


class Adasyn(BaseUpsampling):
    def __init__(self, sampling_strategy):
        self.upsampling = ups.ADASYN(sampling_strategy=sampling_strategy)

    def fit_resample(self, X, y):
        return self.upsampling.fit_resample(X, y)


def create_upsampling(sampling_strategy):
    match CONFIG.UPSAMPLING.upsampling_name:
        case "none":
            return NoneUpsampling()
        case "random":
            return Random(sampling_strategy=sampling_strategy)
        case "smote":
            return Smote(sampling_strategy=sampling_strategy)
        case "borderline_smote":
            return BorderlineSmote(sampling_strategy=sampling_strategy)
        case "svm_smote":
            return SvmSmote(sampling_strategy=sampling_strategy)
        case "kmeans_smote":
            return KMeansSmote(
                    sampling_strategy=sampling_strategy,
                    cluster_balance_threshold=CONFIG.UPSAMPLING.kmeans_cluster_balance_threshold,
                    n_init=CONFIG.UPSAMPLING.kmeans_n_init,
                )
        case "adasyn":
            return Adasyn(sampling_strategy=sampling_strategy)
        case _:
            raise ValueError("Unsupported upsampling name")
