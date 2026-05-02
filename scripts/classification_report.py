"""shared classification report canonical utility의 compatibility wrapper."""

from __future__ import annotations

import shared.src.domain.services.classification_report as _classification_report

safe_divide = _classification_report.safe_divide
build_confusion_matrix = _classification_report.build_confusion_matrix
summarize_per_category = _classification_report.summarize_per_category
render_confusion_table = _classification_report.render_confusion_table
render_per_category_table = _classification_report.render_per_category_table
