from src.evaluation.metrics import (
    median_heuristic_gamma,
    compute_mmd,
    evaluate_mmd,
    physics_filter,
    physics_pass_rate,
    compute_wasserstein_per_feature,
    compute_js_divergence_per_feature,
    compute_autocorr_fidelity,
    save_metrics_json,
    compute_all_metrics,
    # Statistical & temporal metrics
    compute_sliced_wasserstein,
    compute_acf_mae,
    compute_correlation_matrix_diff,
    compute_and_save_statistical_metrics,
)
from src.evaluation.visualization import (
    plot_kde,
    plot_tsne,
    plot_pca,
    plot_training_curves,
    plot_soh_comparison,
    plot_physics_pass_rate,
    plot_autocorr_comparison,
    plot_sequence_samples,
    plot_all,
    plot_correlation_heatmaps,
)
from src.evaluation.downstream_soh import run_downstream_soh

__all__ = [
    # Metrics — core
    "median_heuristic_gamma",
    "compute_mmd",
    "evaluate_mmd",
    "physics_filter",
    # Metrics — distribution
    "physics_pass_rate",
    "compute_wasserstein_per_feature",
    "compute_js_divergence_per_feature",
    "compute_autocorr_fidelity",
    "save_metrics_json",
    "compute_all_metrics",
    # Metrics — statistical & temporal
    "compute_sliced_wasserstein",
    "compute_acf_mae",
    "compute_correlation_matrix_diff",
    "compute_and_save_statistical_metrics",
    # Visualization — core
    "plot_kde",
    "plot_tsne",
    "plot_pca",
    "plot_all",
    # Visualization — extended
    "plot_training_curves",
    "plot_soh_comparison",
    "plot_physics_pass_rate",
    "plot_autocorr_comparison",
    "plot_sequence_samples",
    "plot_correlation_heatmaps",
    # Downstream task
    "run_downstream_soh",
]
