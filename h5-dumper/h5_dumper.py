#!/usr/bin/env python3
"""Dump a ROOT TTree into HDF5 tables using a YAML configuration."""

from __future__ import annotations

import argparse
import ast
import glob
import keyword
import re
from pathlib import Path

import h5py
import numpy as np
import uproot
import yaml


def _strip_affixes(name: str, prefix: str, suffix: str) -> str:
    if prefix and name.startswith(prefix):
        name = name[len(prefix) :]
    if suffix and name.endswith(suffix):
        name = name[: -len(suffix)]
    return name


def _build_canonical_branch_map(tree: uproot.behaviors.TTree.TTree, prefix: str, suffix: str) -> dict[str, str]:
    canonical_to_branch: dict[str, str] = {}
    for branch in tree.keys():
        if not isinstance(branch, str):
            continue
        canonical = branch
        parent, separator, child = branch.partition("/")
        if separator and child.startswith(f"{parent}."):
            canonical = child
        canonical = _strip_affixes(canonical, prefix, suffix)
        if canonical in canonical_to_branch and canonical_to_branch[canonical] != branch:
            raise ValueError(
                f"Ambiguous branch name after stripping affixes: '{canonical}' maps to both "
                f"'{canonical_to_branch[canonical]}' and '{branch}'."
            )
        canonical_to_branch[canonical] = branch
    return canonical_to_branch


def _is_vector_array(values: np.ndarray) -> bool:
    return values.dtype == object or values.ndim > 1


def _as_1d(values: object) -> np.ndarray:
    if values is None:
        return np.array([], dtype=np.float32)
    return np.asarray(values).reshape(-1)


def _scalar_dtype_and_values(values: np.ndarray) -> tuple[np.dtype, np.ndarray]:
    if np.issubdtype(values.dtype, np.bool_):
        return np.int8, values.astype(np.int8)
    if np.issubdtype(values.dtype, np.integer):
        return np.int64, values.astype(np.int64)
    if np.issubdtype(values.dtype, np.floating):
        return np.float32, values.astype(np.float32)
    raise TypeError(f"Unsupported scalar dtype: {values.dtype}")


def _column_dtype(col_config: object, sample_dtype: np.dtype) -> np.dtype:
    if isinstance(col_config, str):
        if np.issubdtype(sample_dtype, np.bool_):
            return np.int8
        if np.issubdtype(sample_dtype, np.integer):
            return np.int64
        return np.float32

    type_name = str(col_config.get("type", "float")).lower()
    if type_name in {"int", "integer"}:
        return np.int64
    if type_name in {"bool", "boolean"}:
        return np.int8
    if type_name in {"float", "double"}:
        return np.float32
    raise TypeError(f"Unsupported configured type '{type_name}' for column '{col_config.get('name')}'.")


def _missing_value(dtype: np.dtype) -> object:
    if np.issubdtype(dtype, np.floating):
        return np.nan
    return 0


def _table_prefixes(table_cfg: dict) -> list[str]:
    prefixes = table_cfg.get("prefixes", [])
    if isinstance(prefixes, str):
        return [prefixes] if prefixes else []
    if not isinstance(prefixes, list) or not all(isinstance(prefix, str) for prefix in prefixes):
        raise TypeError(f"Prefixes for table '{table_cfg.get('name')}' must be a string or list of strings.")
    return prefixes


def _table_separator(table_cfg: dict) -> str:
    separator = table_cfg.get("separator", "_")
    if not isinstance(separator, str):
        raise TypeError(f"Separator for table '{table_cfg.get('name')}' must be a string.")
    return separator


def _compile_condition(
    condition: str,
    prefix: str,
    separator: str,
    canonical_to_branch: dict[str, str],
    column_name: str,
) -> tuple[object, dict[str, str]]:
    try:
        expression = ast.parse(condition, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid condition '{condition}' for custom column '{column_name}'.") from exc

    allowed_nodes = (
        ast.Expression,
        ast.BoolOp,
        ast.And,
        ast.Or,
        ast.UnaryOp,
        ast.UAdd,
        ast.USub,
        ast.Not,
        ast.BinOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.Compare,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.Call,
        ast.Name,
        ast.Load,
        ast.Constant,
    )
    for node in ast.walk(expression):
        if not isinstance(node, allowed_nodes):
            raise ValueError(f"Unsupported syntax in condition '{condition}' for custom column '{column_name}'.")
        if isinstance(node, ast.Call) and (
            not isinstance(node.func, ast.Name) or node.func.id != "abs" or node.keywords
        ):
            raise ValueError(f"Only abs() calls are supported in condition '{condition}'.")

    variable_names = {
        node.id
        for node in ast.walk(expression)
        if isinstance(node, ast.Name) and node.id != "abs"
    }
    branches: dict[str, str] = {}
    for variable_name in variable_names:
        candidates = [f"{prefix}{separator}{variable_name}", variable_name]
        branch_name = next((canonical_to_branch[name] for name in candidates if name in canonical_to_branch), None)
        if branch_name is None:
            raise KeyError(
                f"Missing branch '{variable_name}' in condition '{condition}' for custom column '{column_name}'."
            )
        branches[variable_name] = branch_name

    return compile(expression, f"<custom column {column_name}>", "eval"), branches


_SELECTION_BRANCH_PATTERN = re.compile(
    r"[A-Za-z_]\w*(?:\[\d+\])?(?:\.[A-Za-z_]\w*(?:\[\d+\])?)*"
)


def _compile_selection_condition(
    condition: str,
    canonical_to_branch: dict[str, str],
    column_name: str,
    prefix: str | None = None,
    separator: str = ".",
) -> tuple[object, dict[str, tuple[str, int | None]]]:
    branches: dict[str, tuple[str, int | None]] = {}

    def replace_branch(match: re.Match[str]) -> str:
        branch_expression = match.group()
        if keyword.iskeyword(branch_expression) or branch_expression in {"True", "False", "None"}:
            return branch_expression
        if branch_expression == "abs" and condition[match.end() :].lstrip().startswith("("):
            return branch_expression

        selectors = list(re.finditer(r"\[(\d+)\]", branch_expression))
        if len(selectors) > 1:
            raise ValueError(f"Only one vector index is supported in cut '{condition}'.")
        if selectors:
            if prefix is not None:
                raise ValueError(f"Vector indices in cuts are only supported for global selection: '{condition}'.")
            selector = selectors[0]
            canonical = branch_expression[: selector.start()] + branch_expression[selector.end() :]
            vector_index = int(selector.group(1))
        else:
            canonical = branch_expression
            vector_index = None

        candidates = [canonical]
        if prefix is not None and "." not in canonical:
            candidates.insert(0, f"{prefix}{separator}{canonical}")
        branch_name = next((canonical_to_branch[name] for name in candidates if name in canonical_to_branch), None)
        if branch_name is None:
            raise KeyError(f"Missing branch '{branch_expression}' in cut '{condition}' for '{column_name}'.")

        variable_name = f"_branch_{len(branches)}"
        branches[variable_name] = (branch_name, vector_index)
        return variable_name

    rewritten = _SELECTION_BRANCH_PATTERN.sub(replace_branch, condition)
    try:
        expression = ast.parse(rewritten, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid cut '{condition}' for '{column_name}'.") from exc

    allowed_nodes = (
        ast.Expression,
        ast.BoolOp,
        ast.And,
        ast.Or,
        ast.UnaryOp,
        ast.UAdd,
        ast.USub,
        ast.Not,
        ast.BinOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.Compare,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.Call,
        ast.Name,
        ast.Load,
        ast.Constant,
    )
    for node in ast.walk(expression):
        if not isinstance(node, allowed_nodes):
            raise ValueError(f"Unsupported syntax in cut '{condition}' for '{column_name}'.")
        if isinstance(node, ast.Call) and (
            not isinstance(node.func, ast.Name)
            or node.func.id != "abs"
            or len(node.args) != 1
            or node.keywords
        ):
            raise ValueError(f"Only abs() calls are supported in cut '{condition}'.")

    return compile(expression, f"<cut {column_name}>", "eval"), branches


def _cut_list(cuts: object, scope: str) -> list[str]:
    if cuts is None:
        return []
    if not isinstance(cuts, list) or not all(isinstance(cut, str) and cut.strip() for cut in cuts):
        raise TypeError(f"Cuts for {scope} must be a list of non-empty strings.")
    return cuts


def _selection_value(
    branch_values: np.ndarray,
    event_index: int,
    vector_index: int | None,
    item_index: int | None = None,
) -> tuple[bool, object]:
    event_values = np.asarray(branch_values[event_index])
    if event_values.ndim > 0 or event_values.dtype == object:
        event_values = _as_1d(event_values)
        index = item_index if item_index is not None else (vector_index if vector_index is not None else 0)
        if index >= len(event_values):
            return False, None
        return True, event_values[index].item() if hasattr(event_values[index], "item") else event_values[index]

    if item_index is not None:
        if item_index > 0:
            return False, None
        return True, event_values.item() if hasattr(event_values, "item") else event_values
    if vector_index is not None:
        raise ValueError("A vector index was specified for a scalar branch in a cut.")
    return True, event_values.item() if hasattr(event_values, "item") else event_values


def _passes_selection(
    compiled_cut: object,
    branches: dict[str, tuple[str, int | None]],
    branch_arrays: dict[str, np.ndarray],
    event_index: int,
    item_index: int | None = None,
) -> bool:
    namespace: dict[str, object] = {"abs": abs}
    for variable_name, (branch_name, vector_index) in branches.items():
        present, value = _selection_value(
            branch_arrays[branch_name],
            event_index,
            vector_index,
            item_index,
        )
        if not present:
            return False
        namespace[variable_name] = value
    return bool(eval(compiled_cut, {"__builtins__": {}}, namespace))


def _global_event_mask(
    tree: uproot.behaviors.TTree.TTree,
    cuts: list[str],
    canonical_to_branch: dict[str, str],
    n_events: int,
) -> np.ndarray:
    if not cuts:
        return np.ones(n_events, dtype=bool)

    compiled_cuts = [
        _compile_selection_condition(cut, canonical_to_branch, "global selection")
        for cut in cuts
    ]
    needed_branches = {
        branch_name
        for _, cut_branches in compiled_cuts
        for branch_name, _ in cut_branches.values()
    }
    branch_arrays = (
        tree.arrays(sorted(needed_branches), library="np", entry_stop=n_events)
        if needed_branches
        else {}
    )
    return np.fromiter(
        (
            all(
                _passes_selection(compiled_cut, cut_branches, branch_arrays, event_index)
                for compiled_cut, cut_branches in compiled_cuts
            )
            for event_index in range(n_events)
        ),
        dtype=bool,
        count=n_events,
    )


def _collection_values(values: np.ndarray, vector_index: int = 0) -> tuple[np.dtype, np.ndarray]:
    event_values = [_as_1d(values[event_index]) for event_index in range(len(values))]
    sample = next((items[0] for items in event_values if len(items) > 0), None)
    sample_dtype = np.asarray([sample]).dtype if sample is not None else np.dtype(np.float32)
    dtype, _ = _scalar_dtype_and_values(np.asarray([sample], dtype=sample_dtype))
    result = np.full(len(event_values), _missing_value(dtype), dtype=dtype)
    for event_index, items in enumerate(event_values):
        if len(items) > vector_index:
            result[event_index] = items[vector_index]
    return dtype, result


def _global_column_config(column: object, table_name: str) -> tuple[str, str, int | None]:
    if isinstance(column, str):
        if not column:
            raise TypeError(f"Global column names in table '{table_name}' must be non-empty strings.")
        column_name = branch_expression = column
    elif isinstance(column, dict) and len(column) == 1:
        column_name, branch_expression = next(iter(column.items()))
        if not isinstance(column_name, str) or not column_name:
            raise TypeError(f"Global column names in table '{table_name}' must be non-empty strings.")
        if not isinstance(branch_expression, str) or not branch_expression:
            raise TypeError(f"Branch expressions for global column '{column_name}' must be non-empty strings.")
    else:
        raise TypeError(
            f"Global columns in table '{table_name}' must be branch names or one-entry "
            "output-name-to-branch mappings."
        )

    match = re.search(r"\[(\d+)\]", branch_expression)
    if match is None:
        if "[" in branch_expression or "]" in branch_expression:
            raise ValueError(f"Invalid vector index in global branch expression '{branch_expression}'.")
        return column_name, branch_expression, None

    if re.search(r"\[(\d+)\]", branch_expression[match.end() :]) or "[" in branch_expression[: match.start()]:
        raise ValueError(f"Only one vector index is supported in '{branch_expression}'.")
    canonical = branch_expression[: match.start()] + branch_expression[match.end() :]
    if "[" in canonical or "]" in canonical:
        raise ValueError(f"Invalid vector index in global branch expression '{branch_expression}'.")
    return column_name, canonical, int(match.group(1))


def _build_global_table(
    tree: uproot.behaviors.TTree.TTree,
    table_cfg: dict,
    canonical_to_branch: dict[str, str],
    n_events: int,
) -> np.ndarray:
    fields: list[tuple[str, np.dtype]] = []
    values_by_col: dict[str, np.ndarray] = {}
    table_name = str(table_cfg.get("name"))
    if "prefixes" in table_cfg or "separator" in table_cfg:
        raise ValueError(
            f"Global table '{table_name}' must declare complete branch names in 'columns'; "
            "'prefixes' and 'separator' are only supported for vector tables."
        )

    for col in table_cfg.get("columns", []):
        col_name, canonical, index = _global_column_config(col, table_name)
        if canonical not in canonical_to_branch:
            raise KeyError(f"Missing global branch '{canonical}'.")
        if col_name in values_by_col:
            raise ValueError(f"Duplicate global column name '{col_name}' in table '{table_name}'.")

        branch_name = canonical_to_branch[canonical]
        values = np.asarray(tree.arrays([branch_name], library="np", entry_stop=n_events)[branch_name])
        if _is_vector_array(values):
            dtype, cast_values = _collection_values(values, index if index is not None else 0)
        else:
            if index is not None:
                raise ValueError(
                    f"Global branch expression '{col_name}: {canonical}[{index}]' selects an index "
                    "from a scalar branch."
                )
            dtype, cast_values = _scalar_dtype_and_values(values)
        fields.append((col_name, dtype))
        values_by_col[col_name] = cast_values

    n_rows = len(next(iter(values_by_col.values()))) if values_by_col else n_events
    out = np.empty(n_rows, dtype=np.dtype(fields))
    for col, _ in fields:
        out[col] = values_by_col[col]
    return out


def _resolve_vector_columns(
    table_cfg: dict,
    canonical_to_branch: dict[str, str],
) -> tuple[list[tuple[str, object]], dict[str, dict[str, str]], dict[str, dict[str, object]]]:
    prefixes = _table_prefixes(table_cfg)
    if not prefixes:
        raise ValueError(f"Table '{table_cfg.get('name')}' must define at least one prefix.")
    separator = _table_separator(table_cfg)

    raw_columns = table_cfg.get("columns", [])
    parsed_columns: list[tuple[str, object]] = []
    resolved: dict[str, dict[str, str]] = {prefix: {} for prefix in prefixes}
    custom_specs: dict[str, dict[str, object]] = {}

    for col in raw_columns:
        if isinstance(col, str):
            col_name = col
            parsed_columns.append((col_name, col))
            for prefix in prefixes:
                canonical = f"{prefix}{separator}{col_name}"
                if canonical not in canonical_to_branch:
                    raise KeyError(
                        f"Missing vector branch '{canonical}' for table '{table_cfg.get('name')}'."
                    )
                resolved[prefix][col_name] = canonical_to_branch[canonical]
        elif isinstance(col, dict):
            col_name = col.get("name")
            if not col_name:
                raise ValueError(f"Table '{table_cfg.get('name')}' has a column without 'name'.")
            values_map = col.get("values")
            if not isinstance(values_map, dict):
                raise ValueError(
                    f"Custom column '{col_name}' in table '{table_cfg.get('name')}' must define a 'values' mapping."
                )

            is_prefix_scoped = all(prefix in values_map for prefix in prefixes)
            if is_prefix_scoped:
                has_branch_rules = any(isinstance(values_map[prefix], dict) for prefix in prefixes)
            else:
                has_branch_rules = False

            if is_prefix_scoped and has_branch_rules:
                resolved_rules: dict[str, list[tuple[str, object]]] = {}
                for prefix in prefixes:
                    prefix_rules = values_map[prefix]
                    if not isinstance(prefix_rules, dict):
                        raise ValueError(
                            f"Custom column '{col_name}' must use either fixed values for all prefixes "
                            f"or branch rules for all prefixes."
                        )

                    branch_rules: list[tuple[str, object]] = []
                    for criterion_name, criterion_value in prefix_rules.items():
                        canonical_candidates = [f"{prefix}{separator}{criterion_name}", str(criterion_name)]
                        resolved_branch = None
                        for candidate in canonical_candidates:
                            if candidate in canonical_to_branch:
                                resolved_branch = canonical_to_branch[candidate]
                                break
                        if resolved_branch is None:
                            raise KeyError(
                                f"Missing criterion branch for custom column '{col_name}': "
                                f"tried {canonical_candidates}"
                            )
                        branch_rules.append((resolved_branch, criterion_value))

                    resolved_rules[prefix] = branch_rules

                custom_specs[col_name] = {
                    "mode": "branch_rules",
                    "rules": resolved_rules,
                }
            elif is_prefix_scoped:
                custom_specs[col_name] = {
                    "mode": "fixed",
                    "values": {prefix: values_map[prefix] for prefix in prefixes},
                }
            else:
                resolved_rules = {}
                for prefix in prefixes:
                    expression_rules: list[tuple[object, dict[str, str], object]] = []
                    for condition, mapped_value in values_map.items():
                        if not isinstance(condition, str):
                            raise TypeError(
                                f"Conditions for custom column '{col_name}' must be strings, got {condition!r}."
                            )
                        compiled_condition, condition_branches = _compile_condition(
                            condition, prefix, separator, canonical_to_branch, col_name
                        )
                        expression_rules.append((compiled_condition, condition_branches, mapped_value))
                    resolved_rules[prefix] = expression_rules
                custom_specs[col_name] = {
                    "mode": "expression_rules",
                    "rules": resolved_rules,
                }
            parsed_columns.append((col_name, col))
        else:
            raise TypeError(f"Unsupported column entry in table '{table_cfg.get('name')}': {col!r}")

    return parsed_columns, resolved, custom_specs


def _evaluate_custom_value(
    custom_spec: dict[str, object],
    prefix: str,
    event_index: int,
    item_index: int,
    branch_arrays: dict[str, np.ndarray],
    default_value: object,
) -> object:
    mode = custom_spec.get("mode")
    if mode == "fixed":
        values = custom_spec.get("values", {})
        return values.get(prefix, default_value)

    if mode == "branch_rules":
        rules_by_prefix = custom_spec.get("rules", {})
        rules = rules_by_prefix.get(prefix, [])

        selected = None
        for branch_name, mapped_value in rules:
            event_values = _as_1d(branch_arrays[branch_name][event_index])
            if item_index < len(event_values) and bool(event_values[item_index]):
                if selected is None:
                    selected = mapped_value
                else:
                    try:
                        selected = max(selected, mapped_value)
                    except TypeError:
                        selected = mapped_value

        if selected is None:
            return default_value
        return selected

    if mode == "expression_rules":
        rules_by_prefix = custom_spec.get("rules", {})
        rules = rules_by_prefix.get(prefix, [])
        namespace = {"abs": abs}
        selected = None
        matched = False
        for condition, variable_branches, mapped_value in rules:
            condition_values = {}
            for variable_name, branch_name in variable_branches.items():
                event_values = _as_1d(branch_arrays[branch_name][event_index])
                if item_index >= len(event_values):
                    break
                condition_values[variable_name] = event_values[item_index].item()
            else:
                if bool(eval(condition, {"__builtins__": {}}, {**namespace, **condition_values})):
                    if not matched:
                        selected = mapped_value
                        matched = True
                    else:
                        try:
                            selected = max(selected, mapped_value)
                        except TypeError:
                            selected = mapped_value

        return selected if matched else default_value

    raise ValueError(f"Unsupported custom column mode: {mode}")


def _build_vector_table(
    tree: uproot.behaviors.TTree.TTree,
    table_cfg: dict,
    canonical_to_branch: dict[str, str],
    n_events: int,
) -> tuple[np.ndarray, list[str]]:
    parsed_columns, resolved, custom_specs = _resolve_vector_columns(table_cfg, canonical_to_branch)
    if not parsed_columns:
        raise ValueError(f"Table '{table_cfg.get('name')}' must define at least one column.")
    prefixes = _table_prefixes(table_cfg)
    max_rows = int(table_cfg.get("n_rows", 0))
    if max_rows <= 0:
        raise ValueError(f"Table '{table_cfg.get('name')}' must define n_rows > 0.")

    cuts = _cut_list(table_cfg.get("cuts"), f"table '{table_cfg.get('name')}'")
    cuts_by_prefix: dict[str, list[tuple[object, dict[str, tuple[str, int | None]]]]] = {}
    for prefix in prefixes:
        cuts_by_prefix[prefix] = [
            _compile_selection_condition(
                cut,
                canonical_to_branch,
                f"table '{table_cfg.get('name')}'",
                prefix,
                _table_separator(table_cfg),
            )
            for cut in cuts
        ]

    needed_branches: set[str] = set()
    for per_prefix in resolved.values():
        needed_branches.update(per_prefix.values())
    for compiled_cuts in cuts_by_prefix.values():
        for _, cut_branches in compiled_cuts:
            needed_branches.update(branch_name for branch_name, _ in cut_branches.values())
    for spec in custom_specs.values():
        if spec.get("mode") in {"branch_rules", "expression_rules"}:
            for rules in spec.get("rules", {}).values():
                for rule in rules:
                    if spec["mode"] == "branch_rules":
                        needed_branches.add(rule[0])
                    else:
                        needed_branches.update(rule[1].values())
    branch_arrays = tree.arrays(sorted(needed_branches), library="np", entry_stop=n_events)

    column_names = [col_name for col_name, _ in parsed_columns]

    field_defs: list[tuple[str, np.dtype]] = []
    for col_name, col_cfg in parsed_columns:
        if isinstance(col_cfg, str):
            sample_branch = next(iter(resolved.values()))[col_name]
            sample = np.asarray(branch_arrays[sample_branch])
            sample_dtype = np.float32
            if sample.dtype == object and len(sample) > 0:
                first = _as_1d(sample[0])
                if first.size > 0:
                    sample_dtype = first.dtype
            elif sample.ndim > 1:
                sample_dtype = sample.dtype
            field_defs.append((col_name, _column_dtype(col_cfg, np.dtype(sample_dtype))))
        else:
            field_defs.append((col_name, _column_dtype(col_cfg, np.float32)))

    out_dtype = np.dtype(field_defs)
    table = np.empty((n_events, max_rows), dtype=out_dtype)
    for col_name, _ in parsed_columns:
        fill_value = _missing_value(out_dtype.fields[col_name][0])
        table[col_name].fill(fill_value)

    for event_index in range(n_events):
        row_index = 0
        for prefix in prefixes:
            if row_index >= max_rows:
                break

            vectors: dict[str, np.ndarray] = {}
            n_entries = 0

            for col_name, col_cfg in parsed_columns:
                if isinstance(col_cfg, dict):
                    continue
                branch_name = resolved[prefix][col_name]
                event_values = _as_1d(branch_arrays[branch_name][event_index])
                vectors[col_name] = event_values
                n_entries = max(n_entries, len(event_values))

            for spec in custom_specs.values():
                if spec.get("mode") == "branch_rules":
                    branch_names = [branch_name for branch_name, _ in spec["rules"].get(prefix, [])]
                elif spec.get("mode") == "expression_rules":
                    branch_names = [
                        branch_name
                        for _, variable_branches, _ in spec["rules"].get(prefix, [])
                        for branch_name in variable_branches.values()
                    ]
                else:
                    branch_names = []
                for branch_name in branch_names:
                    event_values = _as_1d(branch_arrays[branch_name][event_index])
                    n_entries = max(n_entries, len(event_values))

            for _, cut_branches in cuts_by_prefix[prefix]:
                for branch_name, _ in cut_branches.values():
                    event_values = _as_1d(branch_arrays[branch_name][event_index])
                    n_entries = max(n_entries, len(event_values))

            for item_index in range(n_entries):
                if any(
                    not _passes_selection(compiled_cut, cut_branches, branch_arrays, event_index, item_index)
                    for compiled_cut, cut_branches in cuts_by_prefix[prefix]
                ):
                    continue

                for col_name, col_cfg in parsed_columns:
                    if isinstance(col_cfg, dict):
                        field_dtype = out_dtype.fields[col_name][0]
                        value = _evaluate_custom_value(
                            custom_specs[col_name],
                            prefix,
                            event_index,
                            item_index,
                            branch_arrays,
                            _missing_value(field_dtype),
                        )
                        table[col_name][event_index, row_index] = value
                        continue

                    vec = vectors[col_name]
                    if item_index < len(vec):
                        value = vec[item_index]
                    else:
                        field_dtype = out_dtype.fields[col_name][0]
                        value = _missing_value(field_dtype)
                    table[col_name][event_index, row_index] = value

                row_index += 1
                if row_index >= max_rows:
                    break

    return table, column_names


def _expand_input_paths(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for pattern in inputs:
        matches = glob.glob(str(Path(pattern).expanduser()), recursive=True)
        if not matches:
            candidate = Path(pattern).expanduser()
            if candidate.exists():
                matches = [str(candidate)]

        for match in sorted(matches):
            path = Path(match).resolve()
            if path.is_file() and path not in seen:
                seen.add(path)
                paths.append(path)

    if not paths:
        raise FileNotFoundError("No input ROOT files matched the provided --input values.")

    return paths


def _table_event_count(table: np.ndarray) -> int:
    if table.shape == ():
        return 0
    return int(table.shape[0])


def _append_table(existing: np.ndarray | None, incoming: np.ndarray) -> np.ndarray:
    if existing is None:
        return incoming
    return np.concatenate((existing, incoming), axis=0)


def _write_output_file(
    output_path: Path,
    tree_name: str,
    table_data: dict[str, np.ndarray],
    table_meta: dict[str, dict[str, object]],
    n_events: int,
    input_files: list[Path],
) -> None:
    with h5py.File(output_path, "w") as h5_file:
        for table_name, data in table_data.items():
            meta = table_meta[table_name]
            dset = h5_file.create_dataset(table_name, data=data, compression="gzip")
            dset.attrs["tree_name"] = tree_name
            dset.attrs["n_events"] = n_events

            if not bool(meta.get("global_table", False)):
                dset.attrs["column_names"] = np.array(meta["column_names"], dtype=h5py.string_dtype("utf-8"))
                dset.attrs["n_rows"] = int(meta["n_rows"])
                dset.attrs["n_vars"] = int(meta["n_vars"])
                dset.attrs["shape_order"] = "n_events,n_rows"

            dset.attrs["source_files"] = np.array([str(p) for p in input_files], dtype=h5py.string_dtype("utf-8"))


def _split_output_path(base_output: Path, index: int, split_enabled: bool) -> Path:
    if not split_enabled:
        return base_output
    stem = base_output.stem
    suffix = base_output.suffix or ".h5"
    return base_output.with_name(f"{stem}_part{index:04d}{suffix}")


def convert_tree(
    input_paths: list[Path],
    output_path: Path,
    config_path: Path,
    n_events: int | None = None,
    split_events: int | None = None,
) -> None:
    output_path = output_path.expanduser().resolve()
    config_path = config_path.expanduser().resolve()

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    tree_name = str(config.get("tree_name"))
    remove_prefix = str(config.get("remove_global_prefix", "") or "")
    remove_suffix = str(config.get("remove_global_suffix", "") or "")
    global_cuts = _cut_list(config.get("cuts"), "global selection")
    tables = list(config.get("tables", []))

    split_limit = None
    if split_events is not None:
        if split_events <= 0:
            raise ValueError("--split-events must be > 0")
        split_limit = int(split_events)

    remaining_events = n_events
    buffered_tables: dict[str, np.ndarray | None] = {}
    table_meta: dict[str, dict[str, object]] = {}
    source_files_used: list[Path] = []
    part_index = 0
    total_written = 0

    for input_path in input_paths:
        if remaining_events is not None and remaining_events <= 0:
            break

        with uproot.open(input_path) as root_file:
            tree = root_file[tree_name]
            total_events_in_file = int(tree.num_entries)
            selected_events = total_events_in_file
            if remaining_events is not None:
                selected_events = min(selected_events, int(remaining_events))

            if selected_events <= 0:
                continue

            canonical_to_branch = _build_canonical_branch_map(tree, remove_prefix, remove_suffix)
            event_mask = _global_event_mask(tree, global_cuts, canonical_to_branch, selected_events)

            for table_cfg in tables:
                table_name = str(table_cfg["name"])
                is_global = bool(table_cfg.get("global_table", False))

                if is_global:
                    data = _build_global_table(tree, table_cfg, canonical_to_branch, selected_events)
                    if _table_event_count(data) != selected_events:
                        raise RuntimeError(
                            f"Global table '{table_name}' has {_table_event_count(data)} events, expected {selected_events}."
                        )
                    if table_name not in table_meta:
                        table_meta[table_name] = {"global_table": True}
                    elif data.dtype != buffered_tables[table_name].dtype:
                        raise RuntimeError(
                            f"Global table '{table_name}' dtype changed across input files: "
                            f"{buffered_tables[table_name].dtype} vs {data.dtype}."
                        )
                else:
                    data, column_names = _build_vector_table(tree, table_cfg, canonical_to_branch, selected_events)
                    expected_shape = (selected_events, int(table_cfg["n_rows"]))
                    if data.shape != expected_shape:
                        raise RuntimeError(
                            f"Vector table '{table_name}' has shape {data.shape}, expected {expected_shape}."
                        )
                    if table_name not in table_meta:
                        table_meta[table_name] = {
                            "global_table": False,
                            "column_names": column_names,
                            "n_rows": int(table_cfg["n_rows"]),
                            "n_vars": len(column_names),
                        }
                    else:
                        prior_meta = table_meta[table_name]
                        if prior_meta["column_names"] != column_names or prior_meta["n_rows"] != int(table_cfg["n_rows"]):
                            raise RuntimeError(
                                f"Vector table '{table_name}' schema changed across input files."
                            )
                        if data.dtype != buffered_tables[table_name].dtype:
                            raise RuntimeError(
                                f"Vector table '{table_name}' dtype changed across input files: "
                                f"{buffered_tables[table_name].dtype} vs {data.dtype}."
                            )

                data = data[event_mask]
                buffered_tables[table_name] = _append_table(buffered_tables.get(table_name), data)

            source_files_used.append(input_path)
            if remaining_events is not None:
                remaining_events -= selected_events

        if not buffered_tables:
            continue

        table_names = list(buffered_tables.keys())
        buffered_events = _table_event_count(buffered_tables[table_names[0]])
        for table_name in table_names[1:]:
            if _table_event_count(buffered_tables[table_name]) != buffered_events:
                raise RuntimeError(f"Buffered table '{table_name}' has inconsistent event count.")

        while split_limit is not None and buffered_events >= split_limit:
            chunk_tables: dict[str, np.ndarray] = {}
            for table_name in table_names:
                data = buffered_tables[table_name]
                chunk_tables[table_name] = data[:split_limit]
                buffered_tables[table_name] = data[split_limit:]

            out_path = _split_output_path(output_path, part_index, split_enabled=True)
            _write_output_file(
                out_path,
                tree_name,
                chunk_tables,
                table_meta,
                split_limit,
                source_files_used,
            )
            total_written += split_limit
            print(f"Wrote {out_path}")
            part_index += 1

            buffered_events = _table_event_count(buffered_tables[table_names[0]])

    if not buffered_tables:
        raise RuntimeError("No events were processed from input files.")

    table_names = list(buffered_tables.keys())
    buffered_events = _table_event_count(buffered_tables[table_names[0]])
    for table_name in table_names[1:]:
        if _table_event_count(buffered_tables[table_name]) != buffered_events:
            raise RuntimeError(f"Buffered table '{table_name}' has inconsistent event count.")

    if buffered_events > 0:
        final_tables: dict[str, np.ndarray] = {name: buffered_tables[name] for name in table_names}
        out_path = _split_output_path(output_path, part_index, split_enabled=(split_limit is not None))
        _write_output_file(
            out_path,
            tree_name,
            final_tables,
            table_meta,
            buffered_events,
            source_files_used,
        )
        total_written += buffered_events
        print(f"Wrote {out_path}")

    print(f"Processed {total_written} events from {len(source_files_used)} input files.")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        "-i",
        nargs="+",
        required=True,
        help="Input ROOT files or glob patterns (space separated)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="output.h5",
        help="Base path to the output HDF5 file",
    )
    parser.add_argument(
        "-c",
        "--config",
        help="Path to the configuration YAML file",
    )
    parser.add_argument(
        "-n",
        "--n-events",
        type=int,
        default=None,
        help="Total number of events to process across all inputs (default: all)",
    )
    parser.add_argument(
        "--split-events",
        type=int,
        default=None,
        help="Maximum number of events per output file; when set, writes numbered .partXXXX files",
    )
    args = parser.parse_args(argv)

    if args.n_events is not None and args.n_events < 0:
        raise ValueError("--n-events must be >= 0")

    input_paths = _expand_input_paths(args.input)
    convert_tree(
        input_paths,
        Path(args.output),
        Path(args.config),
        n_events=args.n_events,
        split_events=args.split_events,
    )


if __name__ == "__main__":
    main()
