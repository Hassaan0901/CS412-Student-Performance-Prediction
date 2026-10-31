"""CS 412 - Student Lifestyle Dataset exploration.

Run this script in the same folder as student_lifestyle_dataset.csv.
"""

import pandas as pd

# Load the student lifestyle dataset.
df = pd.read_csv("student_lifestyle_dataset.csv")

# Check the dataset's size and column names.
print("Number of student records:", len(df))
print("Number of columns:", len(df.columns))
print("\nColumn names:")
for column in df.columns:
    print("-", column)

# Check the data for missing values and duplicates.
print("\nTotal missing values:", df.isnull().sum().sum())
print("Duplicate rows:", df.duplicated().sum())
print("Unique student IDs:", df["Student_ID"].nunique())

# Show basic GPA statistics.
print("\nAverage GPA:", round(df["GPA"].mean(), 2))
print("Minimum GPA:", round(df["GPA"].min(), 2))
print("Maximum GPA:", round(df["GPA"].max(), 2))

# Check the stress-level categories; encoding is left for the model work.
print("\nStress-level categories:")
print(df["Stress_Level"].value_counts().to_string())

# Show a summary table matching Section 3.1 of the intermediate report.
summary = pd.DataFrame({
    "Dataset information": [
        "Total student records", "Total columns", "Missing values",
        "Duplicate rows", "Unique student IDs", "Average GPA",
        "Minimum GPA", "Maximum GPA",
    ],
    "Result": [
        len(df), len(df.columns), df.isnull().sum().sum(),
        df.duplicated().sum(), df["Student_ID"].nunique(),
        round(df["GPA"].mean(), 2), round(df["GPA"].min(), 2),
        round(df["GPA"].max(), 2),
    ],
})
print("\nDataset summary table:")
print(summary.to_string(index=False))
