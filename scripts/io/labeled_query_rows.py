"""shared labeled query row canonical contract의 compatibility wrapper."""

from __future__ import annotations

import shared.src.contracts.labeled_query_row_contracts as _contracts

LabeledQueryRow = _contracts.LabeledQueryRow
load_labeled_query_rows = _contracts.load_labeled_query_rows
dump_labeled_query_rows = _contracts.dump_labeled_query_rows
group_labeled_query_rows_by_label = _contracts.group_labeled_query_rows_by_label
count_labeled_query_rows_by_label = _contracts.count_labeled_query_rows_by_label
