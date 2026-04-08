from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, RandomizedSearchCV, StratifiedKFold


def normalize_site_group(site_id):
    site_text = str(site_id).strip().upper()
    return site_text[:5] if site_text else 'UNKNOWN'


@dataclass(frozen=True)
class CrossValidationConfig:
    use_site_grouped_cv: bool = True
    optimize_hyperparameter_search: bool = False
    outer_random_splits: int = 5
    random_state: int = 42
    search_iterations: int = 20
    fixed_hyperparameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CrossValidationResult:
    threshold: float
    consensus_params: Optional[dict[str, Any]]
    metrics: dict[str, Any]


@dataclass(frozen=True)
class FoldSplit:
    fold_index: int
    train_idx: np.ndarray
    validation_idx: np.ndarray
    label: str


class EnsembleCrossValidator:
    def __init__(
        self,
        config: CrossValidationConfig,
        param_dist,
        default_threshold,
        build_preprocessor: Callable[..., Any],
        build_search_model: Callable[..., Any],
        fit_ensemble: Callable[..., Any],
        predict_probabilities: Callable[..., Any],
    ):
        self.config = config
        self.param_dist = param_dist
        self.default_threshold = default_threshold
        self.build_preprocessor = build_preprocessor
        self.build_search_model = build_search_model
        self.fit_ensemble = fit_ensemble
        self.predict_probabilities = predict_probabilities

    def run(
        self,
        features,
        labels,
        feature_indices,
        modality_presence_indices,
        categorical_indices=None,
        site_groups=None,
    ):
        labels = np.asarray(labels, dtype=np.int32)
        if self.config.use_site_grouped_cv:
            return self._run_grouped_cv(
                features,
                labels,
                site_groups,
                feature_indices,
                modality_presence_indices,
                categorical_indices=categorical_indices,
            )

        return self._run_random_cv(
            features,
            labels,
            feature_indices,
            modality_presence_indices,
            categorical_indices=categorical_indices,
        )

    def _run_grouped_cv(
        self,
        features,
        labels,
        site_groups,
        feature_indices,
        modality_presence_indices,
        categorical_indices=None,
    ):
        site_groups = np.asarray(site_groups)
        unique_sites = np.unique(site_groups)
        if unique_sites.size < 2:
            return CrossValidationResult(
                threshold=self.default_threshold,
                consensus_params=None,
                metrics={
                    'skipped': True,
                    'cv_strategy': 'grouped_by_site',
                    'reason': 'Not enough hospitals to run grouped cross-validation.',
                    'site_groups': unique_sites.tolist(),
                },
            )

        classes = np.unique(labels)
        if len(classes) != 2:
            return CrossValidationResult(
                threshold=self.default_threshold,
                consensus_params=None,
                metrics={
                    'skipped': True,
                    'cv_strategy': 'grouped_by_site',
                    'reason': 'Need both classes to run grouped cross-validation.',
                    'site_groups': unique_sites.tolist(),
                },
            )

        group_cv = LeaveOneGroupOut()
        split_plan = []
        for fold_idx, (train_idx, val_idx) in enumerate(group_cv.split(features, labels, groups=site_groups), start=1):
            held_out_site = normalize_site_group(site_groups[val_idx][0])
            split_plan.append(FoldSplit(
                fold_index=fold_idx,
                train_idx=train_idx,
                validation_idx=val_idx,
                label=held_out_site,
            ))

        consensus, selected_params_per_fold = self._select_consensus_params(
            split_plan,
            features,
            labels,
            feature_indices,
            categorical_indices=categorical_indices,
            site_groups=site_groups,
            label_prefix='held-out hospital',
        )
        fold_metrics, oof_probabilities = self._evaluate_with_fixed_params(
            split_plan,
            features,
            labels,
            feature_indices,
            modality_presence_indices,
            categorical_indices=categorical_indices,
            consensus_params=consensus,
            label_prefix='held-out hospital',
            extra_metric_fields=lambda split: {'held_out_site': split.label},
        )

        return self._finalize_result(
            consensus_params=consensus,
            labels=labels,
            oof_probabilities=oof_probabilities,
            best_params_per_fold=selected_params_per_fold,
            fold_metrics=fold_metrics,
            metadata={
                'cv_strategy': 'grouped_by_site',
                'n_splits': int(len(split_plan)),
                'site_groups': unique_sites.tolist(),
            },
        )

    def _run_random_cv(
        self,
        features,
        labels,
        feature_indices,
        modality_presence_indices,
        categorical_indices=None,
    ):
        outer_cv, n_splits = self._build_stratified_splitter(labels, self.config.outer_random_splits)
        if outer_cv is None:
            return CrossValidationResult(
                threshold=self.default_threshold,
                consensus_params=None,
                metrics={
                    'skipped': True,
                    'cv_strategy': 'random_stratified',
                    'reason': 'Not enough samples per class to run random stratified cross-validation.',
                    'requested_n_splits': int(self.config.outer_random_splits),
                },
            )

        split_plan = [
            FoldSplit(
                fold_index=fold_idx,
                train_idx=train_idx,
                validation_idx=val_idx,
                label='random stratified split',
            )
            for fold_idx, (train_idx, val_idx) in enumerate(outer_cv.split(features, labels), start=1)
        ]
        consensus, selected_params_per_fold = self._select_consensus_params(
            split_plan,
            features,
            labels,
            feature_indices,
            categorical_indices=categorical_indices,
            site_groups=None,
            label_prefix='random split',
        )
        fold_metrics, oof_probabilities = self._evaluate_with_fixed_params(
            split_plan,
            features,
            labels,
            feature_indices,
            modality_presence_indices,
            categorical_indices=categorical_indices,
            consensus_params=consensus,
            label_prefix='random split',
        )

        return self._finalize_result(
            consensus_params=consensus,
            labels=labels,
            oof_probabilities=oof_probabilities,
            best_params_per_fold=selected_params_per_fold,
            fold_metrics=fold_metrics,
            metadata={
                'cv_strategy': 'random_stratified',
                'n_splits': int(n_splits),
            },
        )

    def _select_consensus_params(
        self,
        split_plan,
        features,
        labels,
        feature_indices,
        categorical_indices=None,
        site_groups=None,
        label_prefix='fold',
    ):
        if not self.config.optimize_hyperparameter_search:
            fixed_params = dict(self.config.fixed_hyperparameters)
            print(f"  Hyperparameter search disabled. Using fixed parameters: {fixed_params}")
            return fixed_params, [
                {
                    'fold': int(split.fold_index),
                    'label': split.label,
                    'params': fixed_params,
                    'source': 'fixed_defaults',
                }
                for split in split_plan
            ]

        selected_params_per_fold = []

        for split in split_plan:
            print(f"  Search fold {split.fold_index}/{len(split_plan)} - {label_prefix} {split.label}")
            X_train = features[split.train_idx]
            y_train = labels[split.train_idx]
            search_site_groups = None if site_groups is None else site_groups[split.train_idx]

            fold_preprocessor = self.build_preprocessor(len(y_train), categorical_indices)
            X_train_proc = np.asarray(fold_preprocessor.fit_transform(X_train), dtype=np.float32)
            remapped_feature_indices = fold_preprocessor.transform_feature_indices(feature_indices)
            print(
                f"    Correlation selector kept {X_train_proc.shape[1]}/{X_train.shape[1]} features"
            )
            fold_best_params = self._search_hyperparams(
                X_train_proc[:, remapped_feature_indices['all']],
                y_train,
                site_groups=search_site_groups,
            )
            print(f"    Best params: {fold_best_params}")
            selected_params_per_fold.append({
                'fold': int(split.fold_index),
                'label': split.label,
                'params': fold_best_params,
                'source': 'search',
            })

        consensus = self._consensus_params([item['params'] for item in selected_params_per_fold])
        print(f"  Consensus hyperparameters: {consensus}")
        return consensus, selected_params_per_fold

    def _evaluate_with_fixed_params(
        self,
        split_plan,
        features,
        labels,
        feature_indices,
        modality_presence_indices,
        categorical_indices=None,
        consensus_params=None,
        label_prefix='fold',
        extra_metric_fields=None,
    ):
        oof_probabilities = np.zeros(len(labels), dtype=np.float32)
        fold_metrics = []

        for split in split_plan:
            print(f"  Eval fold {split.fold_index}/{len(split_plan)} - {label_prefix} {split.label}")
            X_train, X_val = features[split.train_idx], features[split.validation_idx]
            y_train, y_val = labels[split.train_idx], labels[split.validation_idx]

            fold_preprocessor = self.build_preprocessor(len(y_train), categorical_indices)
            X_train_proc = np.asarray(fold_preprocessor.fit_transform(X_train), dtype=np.float32)
            remapped_feature_indices = fold_preprocessor.transform_feature_indices(feature_indices)
            print(
                f"    Correlation selector kept {X_train_proc.shape[1]}/{X_train.shape[1]} features"
            )
            fold_models = self.fit_ensemble(
                X_train_proc,
                y_train,
                remapped_feature_indices,
                consensus_params=consensus_params,
            )
            fold_bundle = {
                'models': fold_models,
                'feature_indices': remapped_feature_indices,
                'modality_presence_indices': modality_presence_indices,
                'preprocessor': fold_preprocessor,
                'threshold': self.default_threshold,
            }

            fold_probabilities = self.predict_probabilities(fold_bundle, X_val)
            oof_probabilities[split.validation_idx] = fold_probabilities

            fold_metric_row = self._compute_evaluation_metrics(
                y_val,
                fold_probabilities,
                threshold=self.default_threshold,
            )
            extra_fields = {} if extra_metric_fields is None else extra_metric_fields(split)
            fold_metric_row = {
                **fold_metric_row,
                'fold': int(split.fold_index),
                'train_size': int(len(split.train_idx)),
                'validation_size': int(len(split.validation_idx)),
                **extra_fields,
            }
            fold_metrics.append(fold_metric_row)
            self._print_metrics(f"    Fold {split.fold_index} metrics:", fold_metric_row)

        return fold_metrics, oof_probabilities

    def _finalize_result(self, consensus_params, labels, oof_probabilities, best_params_per_fold, fold_metrics, metadata):
        consensus = dict(consensus_params or {})

        fold_metric_summary = self._summarize_metrics(fold_metrics)
        self._print_metric_summary("  Mean fold metrics:", fold_metric_summary)

        oof_default_metrics = self._compute_evaluation_metrics(
            labels,
            oof_probabilities,
            threshold=self.default_threshold,
        )
        self._print_metrics("  OOF metrics before calibration:", oof_default_metrics)

        threshold = self._best_threshold(oof_probabilities, labels)
        oof_calibrated_metrics = self._compute_evaluation_metrics(
            labels,
            oof_probabilities,
            threshold=threshold,
        )
        self._print_metrics("  OOF metrics after calibration:", oof_calibrated_metrics)

        metrics = {
            'skipped': False,
            **metadata,
            'hyperparameter_optimization_enabled': bool(self.config.optimize_hyperparameter_search),
            'fixed_hyperparameters': dict(self.config.fixed_hyperparameters),
            'selected_params_per_fold': best_params_per_fold,
            'fold_metrics': fold_metrics,
            'fold_metric_summary': fold_metric_summary,
            'oof_default_threshold_metrics': oof_default_metrics,
            'oof_calibrated_metrics': oof_calibrated_metrics,
        }
        return CrossValidationResult(
            threshold=threshold,
            consensus_params=consensus,
            metrics=metrics,
        )

    def _search_hyperparams(self, X_train, y_train, site_groups=None):
        inner_cv, fit_kwargs = self._build_inner_cv(y_train, site_groups)
        if inner_cv is None:
            return {}

        search = RandomizedSearchCV(
            estimator=self.build_search_model(y_train),
            param_distributions=self.param_dist,
            n_iter=self.config.search_iterations,
            scoring='f1',
            cv=inner_cv,
            random_state=self.config.random_state,
            n_jobs=-1,
            refit=False,
        )
        search.fit(X_train, y_train, **fit_kwargs)
        return search.best_params_

    def _build_inner_cv(self, y_train, site_groups=None):
        fit_kwargs = {}

        if self.config.use_site_grouped_cv and site_groups is not None:
            site_groups = np.asarray(site_groups)
            unique_groups = np.unique(site_groups)
            if unique_groups.size >= 2:
                fit_kwargs['groups'] = site_groups
                return LeaveOneGroupOut(), fit_kwargs

        inner_cv, _ = self._build_stratified_splitter(y_train, self.config.outer_random_splits)
        return inner_cv, fit_kwargs

    def _build_stratified_splitter(self, labels, requested_splits):
        classes, class_counts = np.unique(labels, return_counts=True)
        if len(classes) != 2:
            return None, 0

        n_splits = min(int(requested_splits), int(np.min(class_counts)))
        if n_splits < 2:
            return None, 0

        return StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=self.config.random_state,
        ), n_splits

    def _best_threshold(self, probabilities, labels):
        thresholds = np.linspace(0, 1, 101)
        best_score = -1.0
        best_value = self.default_threshold

        for threshold in thresholds:
            predictions = (probabilities >= threshold).astype(np.int32)
            score = f1_score(labels, predictions, zero_division=0)
            if score > best_score:
                best_score = score
                best_value = float(threshold)

        return best_value

    def _safe_auroc(self, labels, probabilities):
        if len(np.unique(labels)) < 2:
            return np.nan
        return float(roc_auc_score(labels, probabilities))

    def _safe_auprc(self, labels, probabilities):
        if len(np.unique(labels)) < 2:
            return np.nan
        return float(average_precision_score(labels, probabilities, pos_label=1))

    def _compute_evaluation_metrics(self, labels, probabilities, threshold):
        labels = np.asarray(labels, dtype=np.int32)
        probabilities = np.asarray(probabilities, dtype=np.float32)
        predictions = (probabilities >= threshold).astype(np.int32)

        return {
            'threshold': float(threshold),
            'auroc': self._safe_auroc(labels, probabilities),
            'auprc': self._safe_auprc(labels, probabilities),
            'accuracy': float(accuracy_score(labels, predictions)),
            'f1': float(f1_score(labels, predictions, zero_division=0)),
            'precision': float(precision_score(labels, predictions, zero_division=0)),
            'recall': float(recall_score(labels, predictions, zero_division=0)),
        }

    def _summarize_metrics(self, metric_rows):
        summary = {}
        metric_names = ('auroc', 'auprc', 'accuracy', 'f1', 'precision', 'recall')

        for metric_name in metric_names:
            metric_values = np.asarray([row[metric_name] for row in metric_rows], dtype=np.float32)
            finite_values = metric_values[np.isfinite(metric_values)]
            if finite_values.size == 0:
                summary[metric_name] = {'mean': None, 'std': None}
                continue

            summary[metric_name] = {
                'mean': float(np.mean(finite_values)),
                'std': float(np.std(finite_values)),
            }

        return summary

    def _consensus_params(self, params_per_fold):
        consensus = {}
        for key in self.param_dist.keys():
            values = [params[key] for params in params_per_fold if key in params]
            if not values:
                continue
            consensus[key] = max(set(values), key=values.count)
        return consensus

    def _format_metric_value(self, value):
        return 'nan' if value is None or not np.isfinite(value) else f'{value:.3f}'

    def _print_metrics(self, prefix, metrics):
        print(
            f"{prefix} AUROC={self._format_metric_value(metrics['auroc'])}, "
            f"AUPRC={self._format_metric_value(metrics['auprc'])}, "
            f"Accuracy={self._format_metric_value(metrics['accuracy'])}, "
            f"F1={self._format_metric_value(metrics['f1'])}, "
            f"Precision={self._format_metric_value(metrics['precision'])}, "
            f"Recall={self._format_metric_value(metrics['recall'])}, "
            f"Threshold={metrics['threshold']:.2f}"
        )

    def _print_metric_summary(self, prefix, summary):
        metric_names = ('auroc', 'auprc', 'accuracy', 'f1', 'precision', 'recall')
        parts = []
        for metric_name in metric_names:
            metric_summary = summary.get(metric_name, {})
            mean_value = self._format_metric_value(metric_summary.get('mean'))
            std_value = self._format_metric_value(metric_summary.get('std'))
            parts.append(f"{metric_name.upper()}={mean_value} +/- {std_value}")
        print(f"{prefix} {', '.join(parts)}")