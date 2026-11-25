"""
Programmer: Felix Cohen, Connor Whyte
Class: CPSC 322, Fall 2026
Final Project
Date: 11/24/2026

Description: General-purpose helper functions reused from our earlier
    programming assignments. Nothing in this file is specific to soccer; the
    project-specific cleaning code lives in data_prep.py.

    Copied unchanged from PA3 (utils.py):
        get_duplicate_groups, count_missing_values, get_frequencies, group_by,
        compute_correlation_coefficient, compute_equal_width_cutoffs,
        discretize_equal_width, get_cutoff_labels
    Copied unchanged from PA5 (mysklearn/myutils.py):
        get_most_frequent
"""
from mysklearn.mypytable import MyPyTable


# ----- Reused from PA3 -----
def get_duplicate_groups(table, key_column_names):
    """Finds every group of rows that share the same key values.

    Parameters:
        table (MyPyTable): table to search for duplicates
        key_column_names (list of str): column names that together identify a row

    Returns:
        MyPyTable: a table containing every row involved in a duplicate (the
            first occurrence and its duplicates), grouped together by key, with
            an extra leading "row index" column
    """
    key_indexes = [table.get_column_index(name) for name in key_column_names]
    rows_by_key = {}
    key_order = []
    for i, row in enumerate(table.data):
        key = tuple(row[index] for index in key_indexes)
        if key not in rows_by_key:
            rows_by_key[key] = []
            key_order.append(key)
        rows_by_key[key].append([i] + row)

    duplicate_rows = []
    for key in key_order:
        if len(rows_by_key[key]) > 1:
            duplicate_rows.extend(rows_by_key[key])
    return MyPyTable(["row index"] + table.column_names, duplicate_rows)


def count_missing_values(table):
    """Counts the number of missing values ("NA") in each column of a table.

    Parameters:
        table (MyPyTable): table to inspect

    Returns:
        dict of str: int: column name mapped to its number of missing values
            (only columns with at least one missing value are included)
    """
    missing_counts = {}
    for col_name in table.column_names:
        count = table.get_column(col_name).count("NA")
        if count > 0:
            missing_counts[col_name] = count
    return missing_counts


def get_frequencies(values):
    """Counts how many times each unique value occurs.

    Parameters:
        values (list of obj): values to count (must be sortable)

    Returns:
        tuple of (list of obj, list of int): the sorted unique values and the
            parallel list of their counts
    """
    counts = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    unique_values = sorted(counts)
    return unique_values, [counts[value] for value in unique_values]


def group_by(table, group_col_name, value_col_name):
    """Groups the values of one column by the values of another column.

    Parameters:
        table (MyPyTable): table to group
        group_col_name (str): column whose values define the groups
        value_col_name (str): column whose values are collected per group

    Returns:
        tuple of (list of obj, list of list of obj): the sorted group names and a
            parallel list of the values in each group
    """
    group_column = table.get_column(group_col_name)
    value_column = table.get_column(value_col_name)
    groups = {}
    for group_value, value in zip(group_column, value_column):
        groups.setdefault(group_value, []).append(value)
    group_names = sorted(groups)
    return group_names, [groups[name] for name in group_names]


def compute_correlation_coefficient(x, y):
    """Computes the Pearson correlation coefficient between two lists of numbers.

    Parameters:
        x (list of numeric): first variable
        y (list of numeric): second variable (parallel to x)

    Returns:
        float: correlation coefficient r between -1 and 1
    """
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    covariance_sum = sum((x_i - x_mean) * (y_i - y_mean) for x_i, y_i in zip(x, y))
    x_spread = sum((x_i - x_mean) ** 2 for x_i in x) ** 0.5
    y_spread = sum((y_i - y_mean) ** 2 for y_i in y) ** 0.5
    return covariance_sum / (x_spread * y_spread)


def compute_equal_width_cutoffs(values, num_bins):
    """Computes the cutoffs that split the range of values into equal-width bins.

    Parameters:
        values (list of numeric): values to discretize
        num_bins (int): number of bins

    Returns:
        list of float: num_bins + 1 cutoffs, from the minimum value to the maximum value
    """
    min_value = min(values)
    bin_width = (max(values) - min_value) / num_bins
    cutoffs = [min_value + i * bin_width for i in range(num_bins)]
    cutoffs.append(max(values))
    return cutoffs


def discretize_equal_width(value, cutoffs):
    """Places a value into a 1-based bin number given equal-width cutoffs.

    Parameters:
        value (numeric): value to discretize
        cutoffs (list of float): bin cutoffs from compute_equal_width_cutoffs()

    Returns:
        int: bin number from 1 to len(cutoffs) - 1

    Notes:
        Bins are half-open [low, high) except the last bin, which also includes
        the maximum value.
    """
    num_bins = len(cutoffs) - 1
    for i in range(num_bins - 1):
        if value < cutoffs[i + 1]:
            return i + 1
    return num_bins


def get_cutoff_labels(cutoffs, decimals=1):
    """Builds a range label (e.g. "9.0--16.8") for each bin defined by cutoffs.

    Parameters:
        cutoffs (list of float): bin cutoffs
        decimals (int): number of decimal places to show

    Returns:
        list of str: one label per bin
    """
    return [f"{cutoffs[i]:.{decimals}f}--{cutoffs[i + 1]:.{decimals}f}"
            for i in range(len(cutoffs) - 1)]


# ----- Reused from PA5 -----
def get_most_frequent(values):
    """Finds the most frequent value in a list
    """
    counts = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    max_count = max(counts.values())
    for value in values:
        if counts[value] == max_count:
            return value
    return None  # only reached if values is empty
