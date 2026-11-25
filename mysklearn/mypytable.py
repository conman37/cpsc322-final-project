"""
Programmer: Felix Cohen
Class: CPSC 322, Fall 2026
Final Project (copied unchanged from PA5)
Date: 11/24/2026

Description: Defines the MyPyTable class, which represents a 2D table of data
    with column names and supports common table operations (column extraction,
    cleaning, summary statistics, and inner/full outer joins).
"""
import copy
import csv
from tabulate import tabulate

class MyPyTable:
    """Represents a 2D table of data with column names.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    def pretty_print(self):
        """Prints the table in a nicely formatted grid structure."""
        print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).
        """
        return len(self.data), len(self.column_names)

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.
        """
        col_index = self.get_column_index(col_identifier)
        column = []
        for row in self.data:
            value = row[col_index]
            if include_missing_values or value != "NA":
                column.append(value)
        return column

    def get_column_index(self, col_identifier):
        """Finds the index of a column given its name or index.
        """
        if isinstance(col_identifier, str):
            if col_identifier not in self.column_names:
                raise ValueError(f"Invalid column name: {col_identifier}")
            return self.column_names.index(col_identifier)
        if isinstance(col_identifier, int) and not isinstance(col_identifier, bool):
            if col_identifier < 0 or col_identifier >= len(self.column_names):
                raise ValueError(f"Invalid column index: {col_identifier}")
            return col_identifier
        raise ValueError(f"Invalid column identifier: {col_identifier}")

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).
        """
        for row in self.data:
            for j, value in enumerate(row):
                try:
                    row[j] = float(value)
                except (ValueError, TypeError):
                    pass

    def drop_rows(self, row_indexes_to_drop):
        """Remove rows from the table data.
        """
        indexes_to_drop = set(row_indexes_to_drop)
        self.data = [row for i, row in enumerate(self.data) if i not in indexes_to_drop]

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.
        """
        with open(filename, "r", encoding="utf-8") as infile:
            reader = csv.reader(infile)
            rows = [row for row in reader if row]
        if rows:
            self.column_names = rows[0]
            self.data = rows[1:]
        else:
            self.column_names = []
            self.data = []
        self.convert_to_numeric()
        return self

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.
        """
        with open(filename, "w", encoding="utf-8", newline="") as outfile:
            writer = csv.writer(outfile, lineterminator="\n")
            writer.writerow(self.column_names)
            writer.writerows(self.data)

    def find_duplicates(self, key_column_names):
        """Returns a list of indexes representing duplicate rows.
        """
        key_indexes = [self.get_column_index(name) for name in key_column_names]
        seen_keys = set()
        duplicate_indexes = []
        for i, row in enumerate(self.data):
            key = tuple(row[index] for index in key_indexes)
            if key in seen_keys:
                duplicate_indexes.append(i)
            else:
                seen_keys.add(key)
        return duplicate_indexes

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA")."""
        self.data = [row for row in self.data if "NA" not in row]

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column
        by the column's original average.
        """
        col_index = self.get_column_index(col_name)
        values = self.get_column(col_index, include_missing_values=False)
        if not values:
            return
        average = sum(values) / len(values)
        for row in self.data:
            if row[col_index] == "NA":
                row[col_index] = average

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.
        """
        stats_header = ["attribute", "min", "max", "mid", "avg", "median"]
        stats_data = []
        for col_name in col_names:
            values = sorted(self.get_column(col_name, include_missing_values=False))
            if not values:
                continue
            num_values = len(values)
            min_value = values[0]
            max_value = values[-1]
            mid_value = (min_value + max_value) / 2
            avg_value = sum(values) / num_values
            middle = num_values // 2
            if num_values % 2 == 0:
                median_value = (values[middle - 1] + values[middle]) / 2
            else:
                median_value = values[middle]
            stats_data.append([col_name, min_value, max_value, mid_value, avg_value, median_value])
        return MyPyTable(stats_header, stats_data)

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined
        with other_table based on key_column_names.
        """
        left_key_indexes = [self.get_column_index(name) for name in key_column_names]
        right_key_indexes = [other_table.get_column_index(name) for name in key_column_names]
        right_other_indexes = [i for i in range(len(other_table.column_names))
                               if i not in right_key_indexes]
        joined_header = self.column_names + [other_table.column_names[i] for i in right_other_indexes]

        joined_data = []
        for left_row in self.data:
            left_key = [left_row[i] for i in left_key_indexes]
            for right_row in other_table.data:
                right_key = [right_row[i] for i in right_key_indexes]
                if left_key == right_key:
                    joined_row = left_row + [right_row[i] for i in right_other_indexes]
                    joined_data.append(joined_row)
        return MyPyTable(joined_header, joined_data)

    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with
        other_table based on key_column_names.
        """
        left_key_indexes = [self.get_column_index(name) for name in key_column_names]
        right_key_indexes = [other_table.get_column_index(name) for name in key_column_names]
        right_other_indexes = [i for i in range(len(other_table.column_names))
                               if i not in right_key_indexes]
        joined_header = self.column_names + [other_table.column_names[i] for i in right_other_indexes]

        joined_data = []
        matched_right_indexes = set()
        for left_row in self.data:
            left_key = [left_row[i] for i in left_key_indexes]
            found_match = False
            for j, right_row in enumerate(other_table.data):
                right_key = [right_row[i] for i in right_key_indexes]
                if left_key == right_key:
                    found_match = True
                    matched_right_indexes.add(j)
                    joined_data.append(left_row + [right_row[i] for i in right_other_indexes])
            if not found_match:
                # left row with no match: pad the right table's attributes
                joined_data.append(left_row + ["NA"] * len(right_other_indexes))

        # right rows with no match
        for j, right_row in enumerate(other_table.data):
            if j in matched_right_indexes:
                continue
            joined_row = ["NA"] * len(self.column_names)
            for left_index, right_index in zip(left_key_indexes, right_key_indexes):
                joined_row[left_index] = right_row[right_index]
            joined_row += [right_row[i] for i in right_other_indexes]
            joined_data.append(joined_row)
        return MyPyTable(joined_header, joined_data)
