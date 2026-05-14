"""SQLite schema for experiment result indexes."""

from __future__ import annotations

SCHEMA_STATEMENTS = (
    """
    create table if not exists experiment_runs (
        run_id text primary key,
        track text not null,
        method_family text not null,
        method_name text not null,
        algorithm_name text,
        selection_slug text,
        labeled_dataset_name text,
        unlabeled_dataset_name text,
        validation_dataset_name text,
        test_dataset_name text,
        seed integer,
        learning_rate real,
        classifier_learning_rate real,
        epochs integer,
        max_train_steps integer,
        train_batch_size integer,
        eval_batch_size integer,
        initial_checkpoint_name text,
        unlabeled_row_count integer,
        train_seconds real,
        training_example_count integer,
        examples_per_second real,
        trainable_param_ratio real,
        created_at text
    )
    """,
    """
    create table if not exists eval_metrics (
        run_id text not null,
        eval_set text not null,
        rows_total integer,
        loss real,
        accuracy_top_1 real,
        macro_precision real,
        macro_recall real,
        macro_f1 real,
        weighted_precision real,
        weighted_recall real,
        weighted_f1 real,
        balanced_accuracy real,
        expected_calibration_error real,
        max_calibration_error real,
        overconfidence_gap real,
        worst_category_f1 text,
        worst_category_f1_value real,
        worst_category_precision real,
        worst_category_recall real,
        mean_true_label_probability real,
        mean_top_1_probability real,
        mean_margin_top1_top2 real,
        correct_top_1 integer,
        primary key (run_id, eval_set)
    )
    """,
    """
    create table if not exists per_class_metrics (
        run_id text not null,
        eval_set text not null,
        category text not null,
        support integer,
        predicted integer,
        correct integer,
        precision real,
        recall real,
        f1 real,
        mean_true_label_probability real,
        mean_top_1_probability real,
        mean_margin_top1_top2 real,
        primary key (run_id, eval_set, category)
    )
    """,
    """
    create table if not exists confusion_matrix_cells (
        run_id text not null,
        eval_set text not null,
        actual_category text not null,
        predicted_category text not null,
        count integer not null,
        primary key (run_id, eval_set, actual_category, predicted_category)
    )
    """,
    """
    create table if not exists epoch_metrics (
        run_id text not null,
        epoch integer not null,
        train_loss real,
        train_sup_loss real,
        train_unsup_loss real,
        train_util_ratio real,
        selection_loss real,
        selection_accuracy_top_1 real,
        selection_macro_f1 real,
        selection_expected_calibration_error real,
        selection_worst_category_f1 text,
        selection_worst_category_f1_value real,
        primary key (run_id, epoch)
    )
    """,
    """
    create table if not exists epoch_per_class_metrics (
        run_id text not null,
        epoch integer not null,
        category text not null,
        support integer,
        predicted integer,
        correct integer,
        precision real,
        recall real,
        f1 real,
        mean_true_label_probability real,
        mean_top_1_probability real,
        mean_margin_top1_top2 real,
        primary key (run_id, epoch, category)
    )
    """,
    "create index if not exists idx_eval_metrics_eval_set on eval_metrics(eval_set)",
    "create index if not exists idx_runs_method on experiment_runs(method_name)",
    "create index if not exists idx_runs_split on experiment_runs(selection_slug)",
    "create index if not exists idx_per_class_category on per_class_metrics(category)",
)
