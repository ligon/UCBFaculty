"""Shared variable-processing utilities for the Feature pipeline.

This module owns TransformSpec, AssignSpec, YAML constructors, and the
process_vars() function used by the Feature class for spec-driven data
loading.
"""
from __future__ import annotations

import importlib
import logging
import warnings
from typing import Any, Callable, List

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Spec types
# ---------------------------------------------------------------------------

class TransformSpec:
    def __init__(self, inputs, func):
        self.inputs = inputs
        self.func = func


class AssignSpec:
    def __init__(self, value):
        self.value = value


# ---------------------------------------------------------------------------
# YAML constructors
# ---------------------------------------------------------------------------

def assign_constructor(loader, node):
    """Parses an !assign tag into an AssignSpec object."""
    value = loader.construct_scalar(node)
    return AssignSpec(value)


def transform_constructor(loader, node):
    """Parses a !transform tag into a TransformSpec object."""
    def _resolve_function(path: str) -> Callable:
        try:
            module_path, function_name = path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            return getattr(module, function_name)
        except (ImportError, AttributeError) as e:
            raise yaml.YAMLError(f"Could not resolve function '{path}': {e}")

    config = loader.construct_sequence(node, deep=True)
    arg_spec, func_path = config
    func = _resolve_function(func_path)

    if isinstance(arg_spec, str):
        return TransformSpec((arg_spec,), func)
    elif isinstance(arg_spec, dict):
        return TransformSpec(arg_spec, func)
    elif isinstance(arg_spec, list):
        return TransformSpec(tuple(arg_spec), func)
    else:
        raise yaml.YAMLError(f"Invalid argument specification: {arg_spec}")


# ---------------------------------------------------------------------------
# Variable processing
# ---------------------------------------------------------------------------

def process_vars(df: pd.DataFrame, vars_config: Any, verbose: bool = False) -> pd.DataFrame:
    """Apply rename / transform / assign operations defined in a _vars block.

    Applies transforms, renames, and assignments declared in YAML specs.
    """
    if not vars_config:
        return df

    processed_df = df.copy()

    if isinstance(vars_config, list):
        for var_config in vars_config:
            processed_df = process_vars(processed_df, var_config, verbose=verbose)
        return processed_df

    if isinstance(vars_config, dict):
        for output_col, definition in vars_config.items():
            # Rename
            if isinstance(definition, str):
                if definition in processed_df.columns:
                    processed_df.rename(columns={definition: output_col}, inplace=True)
                else:
                    processed_df[output_col] = pd.NA
                    if verbose:
                        warnings.warn(
                            f"Column '{definition}' missing for '{output_col}', filling with NA"
                        )
            # Transform
            elif isinstance(definition, TransformSpec):
                input_cols = definition.inputs if isinstance(definition.inputs, tuple) else (definition.inputs,)
                missing = [col for col in input_cols if col not in processed_df.columns]
                if missing:
                    processed_df[output_col] = pd.NA
                    if verbose:
                        warnings.warn(
                            f"Column(s) {missing} missing for transform into '{output_col}', filling with NA"
                        )
                else:
                    processed_df[output_col] = processed_df[list(input_cols)].apply(definition.func, axis=1)
            # Assign
            elif isinstance(definition, AssignSpec):
                processed_df[output_col] = definition.value

    return processed_df


def get_vars_outputs(vars_config: Any) -> List[str]:
    """Extract output variable names from a _vars block."""
    if isinstance(vars_config, dict):
        return list(vars_config.keys())
    elif isinstance(vars_config, str):
        return [vars_config]
    elif isinstance(vars_config, (list, tuple)):
        if len(vars_config) and isinstance(vars_config[0], dict):
            return [list(d.keys())[0] for d in vars_config]
        else:
            return list(vars_config)
    return []


def get_index_columns(index_config: Any) -> List[str]:
    """Extract column names from an _index block."""
    if isinstance(index_config, str):
        return [index_config]
    elif isinstance(index_config, (list, tuple)):
        return list(index_config)
    return []


# ---------------------------------------------------------------------------
# Raw DataFrame loading
# ---------------------------------------------------------------------------

def load_source_dataframe(filespec: Any, sheet: str | None = None) -> pd.DataFrame:
    """Load a raw DataFrame from a source file specification.

    Parameters
    ----------
    filespec : str or list
        Either a filename string or ``[filename, sheet]``.
    sheet : str or None
        Explicit sheet override (used e.g. for sheets_are_years sources).
    """
    from . import rawdata  # local import to avoid circular

    if isinstance(filespec, (list, tuple)):
        fn, sh = filespec
        return rawdata.get_dataframe(fn, sheet=sh)
    if sheet:
        return rawdata.get_dataframe(filespec, sheet=sheet)
    df = rawdata.get_dataframe(filespec)
    if isinstance(df, dict):
        raise ValueError(f"{filespec} has data in {','.join(df.keys())}.  Specify one.")
    return df
