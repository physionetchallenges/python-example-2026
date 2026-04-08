import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

from .config import FEATURE_CORRELATION_THRESHOLD


DEFAULT_KNN_NEIGHBORS = 5


def build_preprocessor(num_samples, categorical_indices=None):
    neighbors = min(DEFAULT_KNN_NEIGHBORS, max(1, num_samples - 1)) if num_samples > 1 else 1
    return CorrelationAwarePreprocessor(
        n_neighbors=neighbors,
        categorical_indices=categorical_indices,
        correlation_threshold=FEATURE_CORRELATION_THRESHOLD,
    )


def get_processed_feature_names(feature_names, preprocessor=None):
    if preprocessor is None:
        return list(feature_names)

    if hasattr(preprocessor, 'get_feature_names_out'):
        return list(preprocessor.get_feature_names_out(feature_names))

    return list(feature_names)


def remap_feature_indices(preprocessor, feature_indices):
    if preprocessor is None or not hasattr(preprocessor, 'transform_feature_indices'):
        return {
            name: np.asarray(indices, dtype=np.int32)
            for name, indices in feature_indices.items()
        }

    return preprocessor.transform_feature_indices(feature_indices)


class CorrelationThresholdSelector:
    def __init__(self, threshold):
        self.threshold = float(threshold)
        self.selected_indices_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        n_samples, n_features = X.shape

        if n_features == 0:
            self.selected_indices_ = np.array([], dtype=np.int32)
            return self

        if n_samples < 2 or n_features == 1:
            self.selected_indices_ = np.arange(n_features, dtype=np.int32)
            return self

        with np.errstate(divide='ignore', invalid='ignore'):
            corr = np.corrcoef(X, rowvar=False)
        corr = np.asarray(corr, dtype=np.float32)
        corr = np.nan_to_num(np.abs(corr), nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(corr, 0.0)

        keep_mask = np.ones(n_features, dtype=bool)
        for index in range(n_features):
            if not keep_mask[index]:
                continue

            correlated_indices = np.where(corr[index, index + 1:] > self.threshold)[0]
            if correlated_indices.size:
                keep_mask[correlated_indices + index + 1] = False

        if not np.any(keep_mask):
            keep_mask[0] = True

        self.selected_indices_ = np.flatnonzero(keep_mask).astype(np.int32)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        if self.selected_indices_ is None:
            raise ValueError('Correlation selector has not been fitted.')
        return X[:, self.selected_indices_]

    def get_feature_names_out(self, input_features=None):
        if self.selected_indices_ is None:
            raise ValueError('Correlation selector has not been fitted.')

        if input_features is None:
            input_features = [f'feature_{index}' for index in self.selected_indices_]

        return np.asarray([input_features[index] for index in self.selected_indices_], dtype=object)


class CorrelationAwarePreprocessor:
    def __init__(self, n_neighbors, categorical_indices, correlation_threshold):
        self.n_neighbors = n_neighbors
        if categorical_indices is None:
            self.categorical_indices = np.array([], dtype=np.int32)
        else:
            self.categorical_indices = np.asarray(categorical_indices, dtype=np.int32)
        self.imputer = KNNImputer(n_neighbors=n_neighbors, keep_empty_features=True)
        self.scaler = StandardScaler()
        self.selector = CorrelationThresholdSelector(correlation_threshold)
        self._numerical_indices = np.array([], dtype=np.int32)

    def _get_numerical_indices(self, n_features):
        all_idx = np.arange(n_features, dtype=np.int32)
        return np.setdiff1d(all_idx, self.categorical_indices)

    def _scale_numerical_columns(self, X_imputed, fit=False):
        X_out = X_imputed.copy()
        if self._numerical_indices.size == 0:
            return X_out

        X_num = X_imputed[:, self._numerical_indices]
        if fit:
            X_num_scaled = np.asarray(self.scaler.fit_transform(X_num), dtype=np.float32)
        else:
            X_num_scaled = np.asarray(self.scaler.transform(X_num), dtype=np.float32)

        X_out[:, self._numerical_indices] = X_num_scaled
        return X_out

    def fit_transform(self, X):
        X = np.asarray(X, dtype=np.float32).copy()
        X[~np.isfinite(X)] = np.nan

        X_imputed = np.asarray(self.imputer.fit_transform(X), dtype=np.float32)
        self._numerical_indices = self._get_numerical_indices(X.shape[1])
        X_out = self._scale_numerical_columns(X_imputed, fit=True)
        self.selector.fit(X_out)
        return np.asarray(self.selector.transform(X_out), dtype=np.float32)

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32).copy()
        X[~np.isfinite(X)] = np.nan
        X_imputed = np.asarray(self.imputer.transform(X), dtype=np.float32)
        X_out = self._scale_numerical_columns(X_imputed, fit=False)
        return np.asarray(self.selector.transform(X_out), dtype=np.float32)

    def transform_feature_indices(self, feature_indices):
        if self.selector.selected_indices_ is None:
            raise ValueError('Preprocessor has not been fitted.')

        index_lookup = {
            int(raw_index): int(processed_index)
            for processed_index, raw_index in enumerate(self.selector.selected_indices_)
        }
        remapped = {}
        for name, indices in feature_indices.items():
            kept_indices = [
                index_lookup[int(raw_index)]
                for raw_index in np.asarray(indices, dtype=np.int32)
                if int(raw_index) in index_lookup
            ]
            remapped[name] = np.asarray(kept_indices, dtype=np.int32)
        return remapped

    def get_feature_names_out(self, input_features=None):
        return self.selector.get_feature_names_out(input_features)