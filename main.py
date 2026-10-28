from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "student_performance.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    # 1. Load the dataset
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Create the data folder and add "
            "student_performance.csv."
        )

    df = pd.read_csv(DATA_PATH)

    print("\n========== DATASET PREVIEW ==========")
    print(df.head())

    # 2. Dataset structure
    print("\n========== DATASET SHAPE ==========")
    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])

    print("\n========== COLUMN DATA TYPES ==========")
    print(df.dtypes)

    # 3. Check data quality
    print("\n========== MISSING VALUES ==========")
    print(df.isnull().sum())

    print("\n========== DUPLICATE ROWS ==========")
    print("Duplicate rows:", df.duplicated().sum())

    # 4. Descriptive statistics
    numeric = df.select_dtypes(include="number")

    print("\n========== DESCRIPTIVE STATISTICS ==========")
    print(numeric.describe().round(2))

    # 5. Calculate each student's average score
    score_columns = [
        "math_score",
        "science_score",
        "english_score"
    ]

    df["average_score"] = (
        df[score_columns].mean(axis=1).round(2)
    )

    print("\n========== SUBJECT-WISE AVERAGES ==========")
    print(df[score_columns].mean().round(2))

    print(
        "\nOverall average score:",
        round(df["average_score"].mean(), 2)
    )

    highest = df.loc[df["average_score"].idxmax()]
    lowest = df.loc[df["average_score"].idxmin()]

    print(
        "Highest average:",
        highest["student_name"],
        highest["average_score"]
    )

    print(
        "Lowest average:",
        lowest["student_name"],
        lowest["average_score"]
    )

    # 6. Correlation analysis
    correlation_columns = [
        "study_hours_per_day",
        "attendance_percent",
        "math_score",
        "science_score",
        "english_score"
    ]

    correlation = df[correlation_columns].corr()

    print("\n========== CORRELATION MATRIX ==========")
    print(correlation.round(2))

    # 7. Save the EDA report
    report_path = OUTPUT_DIR / "eda_report.txt"

    with report_path.open("w", encoding="utf-8") as file:
        file.write("CODEALPHA - EDA REPORT\n")
        file.write("=" * 35 + "\n\n")

        file.write(f"Rows: {df.shape[0]}\n")
        file.write(
            f"Original columns: {df.shape[1] - 1}\n\n"
        )

        file.write("Missing values:\n")
        file.write(df.isnull().sum().to_string())
        file.write("\n\n")

        file.write(
            f"Duplicate rows: {df.duplicated().sum()}\n\n"
        )

        file.write("Descriptive statistics:\n")
        file.write(numeric.describe().round(2).to_string())
        file.write("\n\n")

        file.write("Average scores by subject:\n")
        file.write(df[score_columns].mean().round(2).to_string())
        file.write("\n\n")

        file.write(
            f"Overall average: "
            f"{df['average_score'].mean():.2f}\n\n"
        )

        file.write("Correlation matrix:\n")
        file.write(correlation.round(2).to_string())

    # 8. Visualize score distribution
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 5))
    sns.histplot(
        data=df,
        x="average_score",
        bins=8,
        kde=True
    )

    plt.title("Distribution of Students' Average Scores")
    plt.xlabel("Average Score")
    plt.ylabel("Number of Students")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "average_score_distribution.png",
        dpi=150
    )
    plt.close()

    # 9. Correlation heatmap
    plt.figure(figsize=(9, 6))

    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        fmt=".2f"
    )

    plt.title("Correlation Between Study, Attendance and Scores")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "correlation_heatmap.png",
        dpi=150
    )
    plt.close()

    # 10. Study hours vs average score
    plt.figure(figsize=(8, 5))

    sns.scatterplot(
        data=df,
        x="study_hours_per_day",
        y="average_score",
        s=75
    )

    sns.regplot(
        data=df,
        x="study_hours_per_day",
        y="average_score",
        scatter=False
    )

    plt.title("Study Hours vs Average Score")
    plt.xlabel("Study Hours per Day")
    plt.ylabel("Average Score")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "study_hours_vs_score.png",
        dpi=150
    )
    plt.close()

    print("\nEDA completed successfully!")
    print("Report and charts saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()