#eda分析模块
from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


TaskType = Literal["auto", "binary_classification", "multiclass_classification", "regression"]


def safe_filename(name: str) -> str:
    """Convert a column name to a safe filename."""
    name = str(name)
    name = re.sub(r"[^\w\-]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name[:80] if name else "unnamed"


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it does not exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def infer_id_cols(df: pd.DataFrame) -> list[str]:
    """Infer possible ID columns from column names."""
    id_names = {
        "id",
        "customerid",
        "customer_id",
        "clientid",
        "client_id",
        "userid",
        "user_id",
        "index",
    }

    id_cols = []
    for col in df.columns:
        col_lower = str(col).lower()
        if col_lower in id_names or col_lower.endswith("_id"):
            id_cols.append(col)

    return id_cols


def build_overview(df: pd.DataFrame) -> pd.DataFrame:
    """Build column-level overview table."""
    rows = []

    for col in df.columns:
        sample_values = df[col].dropna().unique()[:5]
        sample_values = ", ".join(map(str, sample_values))

        rows.append(
            {
                "column": col,
                "dtype": str(df[col].dtype),
                "n_missing": int(df[col].isna().sum()),
                "missing_rate": float(df[col].isna().mean()),
                "n_unique": int(df[col].nunique(dropna=True)),
                "sample_values": sample_values,
            }
        )

    return pd.DataFrame(rows)


def split_feature_types(
    train: pd.DataFrame,
    target_col: str | None,
    id_cols: Sequence[str] | None = None,
    low_cardinality_threshold: int = 20,
) -> dict[str, list[str]]:
    """Split columns into ID, continuous numeric, categorical-like columns."""
    id_cols = list(id_cols or [])

    feature_cols = train.columns.tolist()
    if target_col is not None and target_col in feature_cols:
        feature_cols.remove(target_col)

    numeric_cols = train[feature_cols].select_dtypes(include=["number", "bool"]).columns.tolist()
    raw_categorical_cols = train[feature_cols].select_dtypes(exclude=["number", "bool"]).columns.tolist()

    low_card_num_cols = [
        col
        for col in numeric_cols
        if col not in id_cols and train[col].nunique(dropna=True) <= low_cardinality_threshold
    ]

    continuous_cols = [
        col
        for col in numeric_cols
        if col not in id_cols and col not in low_card_num_cols
    ]

    categorical_like_cols = raw_categorical_cols + low_card_num_cols

    return {
        "feature_cols": feature_cols,
        "id_cols": id_cols,
        "numeric_cols": numeric_cols,
        "raw_categorical_cols": raw_categorical_cols,
        "low_card_num_cols": low_card_num_cols,
        "continuous_cols": continuous_cols,
        "categorical_like_cols": categorical_like_cols,
    }


def to_binary_numeric(series: pd.Series) -> pd.Series:
    """Convert a binary target to numeric 0/1 where possible."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    mapper = {
        "yes": 1,
        "y": 1,
        "true": 1,
        "t": 1,
        "1": 1,
        "churn": 1,
        "no": 0,
        "n": 0,
        "false": 0,
        "f": 0,
        "0": 0,
        "not churn": 0,
    }

    return series.astype(str).str.strip().str.lower().map(mapper)


def infer_task_type(train: pd.DataFrame, target_col: str | None) -> str:
    """Infer task type from target column."""
    if target_col is None or target_col not in train.columns:
        return "unknown"

    target = train[target_col]
    n_unique = target.nunique(dropna=True)

    if pd.api.types.is_numeric_dtype(target):
        if n_unique == 2:
            return "binary_classification"
        if n_unique <= 20:
            return "multiclass_classification"
        return "regression"

    if n_unique == 2:
        return "binary_classification"

    return "multiclass_classification"


def save_target_distribution(
    train: pd.DataFrame,
    target_col: str,
    table_dir: Path,
    figure_dir: Path,
) -> pd.DataFrame:
    """Save target distribution table and figure."""
    target_dist = train[target_col].value_counts(dropna=False).rename("count").to_frame()
    target_dist["ratio"] = target_dist["count"] / len(train)

    target_dist.to_csv(table_dir / "target_distribution.csv", encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(6, 4))
    x_labels = target_dist.index.astype(str)
    ax.bar(x_labels, target_dist["count"])
    ax.set_title(f"Target Distribution: {target_col}")
    ax.set_xlabel(target_col)
    ax.set_ylabel("Count")

    for i, value in enumerate(target_dist["count"]):
        ax.text(i, value, str(value), ha="center", va="bottom")

    fig.tight_layout()
    fig.savefig(figure_dir / "target_distribution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    return target_dist


def save_missing_summary(
    train: pd.DataFrame,
    test: pd.DataFrame | None,
    table_dir: Path,
) -> pd.DataFrame:
    """Save missing value summary."""
    rows = []

    for col in train.columns:
        row = {
            "column": col,
            "train_missing_count": int(train[col].isna().sum()),
            "train_missing_rate": float(train[col].isna().mean()),
        }

        if test is not None and col in test.columns:
            row["test_missing_count"] = int(test[col].isna().sum())
            row["test_missing_rate"] = float(test[col].isna().mean())
        else:
            row["test_missing_count"] = np.nan
            row["test_missing_rate"] = np.nan

        rows.append(row)

    missing_summary = pd.DataFrame(rows)
    missing_summary = missing_summary.sort_values(
        ["train_missing_rate", "test_missing_rate"],
        ascending=False,
    )

    missing_summary.to_csv(table_dir / "missing_summary.csv", index=False, encoding="utf-8-sig")
    return missing_summary


def save_duplicate_summary(
    train: pd.DataFrame,
    test: pd.DataFrame | None,
    id_cols: Sequence[str],
    table_dir: Path,
) -> pd.DataFrame:
    """Save duplicated row and duplicated ID summary."""
    rows = [
        {
            "item": "train_duplicated_rows",
            "count": int(train.duplicated().sum()),
        }
    ]

    if test is not None:
        rows.append(
            {
                "item": "test_duplicated_rows",
                "count": int(test.duplicated().sum()),
            }
        )

    for id_col in id_cols:
        if id_col in train.columns:
            rows.append(
                {
                    "item": f"train_duplicated_{id_col}",
                    "count": int(train[id_col].duplicated().sum()),
                }
            )

        if test is not None and id_col in test.columns:
            rows.append(
                {
                    "item": f"test_duplicated_{id_col}",
                    "count": int(test[id_col].duplicated().sum()),
                }
            )

    duplicate_summary = pd.DataFrame(rows)
    duplicate_summary.to_csv(table_dir / "duplicate_summary.csv", index=False, encoding="utf-8-sig")
    return duplicate_summary


def save_numeric_describe(
    train: pd.DataFrame,
    continuous_cols: Sequence[str],
    table_dir: Path,
) -> pd.DataFrame:
    """Save numeric descriptive statistics."""
    if not continuous_cols:
        return pd.DataFrame()

    numeric_describe = train[list(continuous_cols)].describe().T
    numeric_describe["missing_rate"] = train[list(continuous_cols)].isna().mean()
    numeric_describe["skew"] = train[list(continuous_cols)].skew(numeric_only=True)
    numeric_describe["kurtosis"] = train[list(continuous_cols)].kurtosis(numeric_only=True)

    numeric_describe.to_csv(table_dir / "numeric_describe.csv", encoding="utf-8-sig")
    return numeric_describe


def save_numeric_distribution_plots(
    train: pd.DataFrame,
    continuous_cols: Sequence[str],
    output_dir: Path,
    max_plot_cols: int = 40,
) -> None:
    """Save histogram plots for numeric columns."""
    ensure_dir(output_dir)

    for col in list(continuous_cols)[:max_plot_cols]:
        data = train[col].dropna()

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(data, bins=30)
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")

        fig.tight_layout()
        fig.savefig(output_dir / f"distribution_{safe_filename(col)}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def save_categorical_distribution_plots(
    train: pd.DataFrame,
    categorical_cols: Sequence[str],
    table_dir: Path,
    output_dir: Path,
) -> pd.DataFrame:
    """Save categorical count tables and bar plots."""
    ensure_dir(output_dir)
    rows = []

    for col in categorical_cols:
        vc = train[col].astype("object").fillna("__MISSING__").value_counts(dropna=False)

        for level, count in vc.items():
            rows.append(
                {
                    "column": col,
                    "level": level,
                    "count": int(count),
                    "ratio": float(count / len(train)),
                }
            )

        plot_data = vc.head(20).sort_values()

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh(plot_data.index.astype(str), plot_data.values)
        ax.set_title(f"Top Categories of {col}")
        ax.set_xlabel("Count")
        ax.set_ylabel(col)

        fig.tight_layout()
        fig.savefig(output_dir / f"count_{safe_filename(col)}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    categorical_summary = pd.DataFrame(rows)

    if not categorical_summary.empty:
        categorical_summary.to_csv(
            table_dir / "categorical_count_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

    return categorical_summary


def save_numeric_by_binary_target(
    train: pd.DataFrame,
    continuous_cols: Sequence[str],
    target_numeric: pd.Series,
    table_dir: Path,
    output_dir: Path,
    max_plot_cols: int = 40,
) -> pd.DataFrame:
    """Save numeric feature summaries and boxplots by binary target."""
    ensure_dir(output_dir)

    rows = []

    temp_train = train.copy()
    temp_train["_target_numeric_eda"] = target_numeric

    for col in continuous_cols:
        temp = temp_train[[col, "_target_numeric_eda"]].dropna()

        if temp.empty:
            continue

        grouped = (
            temp.groupby("_target_numeric_eda")[col]
            .agg(count="count", mean="mean", median="median", std="std", min="min", max="max")
            .reset_index()
        )
        grouped.insert(0, "column", col)
        rows.append(grouped)

        if col in list(continuous_cols)[:max_plot_cols]:
            classes = sorted(temp["_target_numeric_eda"].dropna().unique())
            data_by_class = [
                temp.loc[temp["_target_numeric_eda"] == cls, col]
                for cls in classes
            ]

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.boxplot(
                data_by_class,
                labels=[str(int(cls)) if float(cls).is_integer() else str(cls) for cls in classes],
                showfliers=False,
            )
            ax.set_title(f"{col} by Target")
            ax.set_xlabel("Target")
            ax.set_ylabel(col)

            fig.tight_layout()
            fig.savefig(output_dir / f"boxplot_{safe_filename(col)}_by_target.png", dpi=200, bbox_inches="tight")
            plt.close(fig)

    if rows:
        result = pd.concat(rows, ignore_index=True)
    else:
        result = pd.DataFrame()

    if not result.empty:
        result.to_csv(table_dir / "numeric_by_target_summary.csv", index=False, encoding="utf-8-sig")

    return result


def save_categorical_by_binary_target(
    train: pd.DataFrame,
    categorical_cols: Sequence[str],
    target_numeric: pd.Series,
    table_dir: Path,
    output_dir: Path,
) -> pd.DataFrame:
    """Save categorical feature target rate tables and plots."""
    ensure_dir(output_dir)

    rows = []
    temp_train = train.copy()
    temp_train["_target_numeric_eda"] = target_numeric

    for col in categorical_cols:
        temp = temp_train[[col, "_target_numeric_eda"]].copy()
        temp[col] = temp[col].astype("object").fillna("__MISSING__")

        summary = (
            temp.groupby(col)["_target_numeric_eda"]
            .agg(count="count", target_rate="mean")
            .reset_index()
        )

        summary.insert(0, "column", col)
        summary = summary.sort_values(["column", "count"], ascending=[True, False])
        rows.append(summary)

        plot_data = summary.sort_values("count", ascending=False).head(20)
        plot_data = plot_data.sort_values("target_rate")

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh(plot_data[col].astype(str), plot_data["target_rate"])
        ax.set_title(f"Target Rate by {col}")
        ax.set_xlabel("Target Rate")
        ax.set_ylabel(col)

        fig.tight_layout()
        fig.savefig(output_dir / f"target_rate_{safe_filename(col)}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    if rows:
        result = pd.concat(rows, ignore_index=True)
    else:
        result = pd.DataFrame()

    if not result.empty:
        result.to_csv(table_dir / "categorical_by_target_summary.csv", index=False, encoding="utf-8-sig")

    return result


def save_correlation_analysis(
    train: pd.DataFrame,
    continuous_cols: Sequence[str],
    target_col: str,
    target_numeric: pd.Series,
    table_dir: Path,
    figure_dir: Path,
) -> pd.DataFrame:
    """Save correlation matrix and target correlation plots."""
    if not continuous_cols:
        return pd.DataFrame()

    corr_data = train[list(continuous_cols)].copy()
    corr_data[target_col] = target_numeric

    corr_matrix = corr_data.corr(numeric_only=True)
    corr_matrix.to_csv(table_dir / "correlation_matrix.csv", encoding="utf-8-sig")

    if target_col not in corr_matrix.columns:
        return pd.DataFrame()

    target_corr = corr_matrix[target_col].drop(target_col).sort_values(
        key=lambda s: s.abs(),
        ascending=False,
    )

    target_corr_df = target_corr.rename("correlation_with_target").to_frame()
    target_corr_df["abs_correlation_with_target"] = target_corr_df["correlation_with_target"].abs()
    target_corr_df.to_csv(table_dir / "target_correlation.csv", encoding="utf-8-sig")

    top_corr = target_corr.head(20).sort_values()

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(top_corr.index, top_corr.values)
    ax.set_title("Top Numeric Correlations with Target")
    ax.set_xlabel("Correlation")
    ax.set_ylabel("Feature")

    fig.tight_layout()
    fig.savefig(figure_dir / "target_correlation_top20.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    selected_cols = [target_col] + target_corr.abs().head(min(20, len(target_corr))).index.tolist()
    heatmap_data = corr_matrix.loc[selected_cols, selected_cols]

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(heatmap_data.values, aspect="auto")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(len(selected_cols)))
    ax.set_yticks(np.arange(len(selected_cols)))
    ax.set_xticklabels(selected_cols, rotation=90)
    ax.set_yticklabels(selected_cols)
    ax.set_title("Correlation Heatmap: Top Features and Target")

    fig.tight_layout()
    fig.savefig(figure_dir / "correlation_heatmap_top_features.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    return target_corr_df


def save_train_test_shift(
    train: pd.DataFrame,
    test: pd.DataFrame | None,
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
    id_cols: Sequence[str],
    table_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Save train-test distribution shift summaries."""
    if test is None:
        return pd.DataFrame(), pd.DataFrame()

    numeric_rows = []

    for col in numeric_cols:
        if col in id_cols or col not in test.columns:
            continue

        train_mean = train[col].mean()
        test_mean = test[col].mean()
        train_std = train[col].std()
        test_std = test[col].std()

        numeric_rows.append(
            {
                "column": col,
                "train_mean": train_mean,
                "test_mean": test_mean,
                "mean_diff": test_mean - train_mean,
                "train_std": train_std,
                "test_std": test_std,
                "std_ratio_test_over_train": test_std / train_std if train_std != 0 else np.nan,
                "train_missing_rate": train[col].isna().mean(),
                "test_missing_rate": test[col].isna().mean(),
                "train_n_unique": train[col].nunique(dropna=True),
                "test_n_unique": test[col].nunique(dropna=True),
            }
        )

    numeric_shift = pd.DataFrame(numeric_rows)

    if not numeric_shift.empty:
        numeric_shift["abs_mean_diff"] = numeric_shift["mean_diff"].abs()
        numeric_shift = numeric_shift.sort_values("abs_mean_diff", ascending=False)
        numeric_shift.to_csv(
            table_dir / "numeric_train_test_shift.csv",
            index=False,
            encoding="utf-8-sig",
        )

    categorical_rows = []

    for col in categorical_cols:
        if col not in test.columns:
            continue

        train_levels = set(train[col].astype("object").dropna().unique())
        test_levels = set(test[col].astype("object").dropna().unique())

        new_levels = test_levels - train_levels
        missing_levels = train_levels - test_levels

        categorical_rows.append(
            {
                "column": col,
                "train_n_levels": len(train_levels),
                "test_n_levels": len(test_levels),
                "n_new_levels_in_test": len(new_levels),
                "new_levels_in_test": ", ".join(map(str, list(new_levels)[:20])),
                "n_missing_levels_in_test": len(missing_levels),
                "missing_levels_in_test": ", ".join(map(str, list(missing_levels)[:20])),
            }
        )

    categorical_shift = pd.DataFrame(categorical_rows)

    if not categorical_shift.empty:
        categorical_shift = categorical_shift.sort_values("n_new_levels_in_test", ascending=False)
        categorical_shift.to_csv(
            table_dir / "categorical_train_test_shift.csv",
            index=False,
            encoding="utf-8-sig",
        )

    return numeric_shift, categorical_shift


def write_eda_report(
    output_dir: Path,
    train: pd.DataFrame,
    test: pd.DataFrame | None,
    target_col: str | None,
    task_type: str,
    id_cols: Sequence[str],
    continuous_cols: Sequence[str],
    categorical_cols: Sequence[str],
    target_dist: pd.DataFrame | None,
    missing_summary: pd.DataFrame,
    target_corr: pd.DataFrame | None,
) -> Path:
    """Write a basic Markdown EDA report."""
    train_rows, train_cols = train.shape

    if test is not None:
        test_text = f"测试集规模为{test.shape[0]}行、{test.shape[1]}列。"
    else:
        test_text = "未提供测试集。"

    if target_col is not None and target_dist is not None and not target_dist.empty:
        target_text_parts = []
        for idx, row in target_dist.iterrows():
            target_text_parts.append(f"{idx}: {int(row['count'])}({row['ratio']:.2%})")
        target_text = "；".join(target_text_parts)
    else:
        target_text = "未提供目标变量，跳过目标变量分布分析。"

    missing_nonzero = missing_summary[
        (missing_summary["train_missing_rate"] > 0)
        | (missing_summary["test_missing_rate"].fillna(0) > 0)
    ]

    if missing_nonzero.empty:
        missing_text = "训练集与测试集未发现缺失值。"
    else:
        top_missing = missing_nonzero.head(10)
        missing_text = (
            "存在缺失值，缺失率较高的字段包括："
            + "、".join(top_missing["column"].astype(str).tolist())
            + "。"
        )

    if target_corr is not None and not target_corr.empty:
        corr_col = "correlation_with_target"
        top_corr_text = "、".join(
            [
                f"{idx}({row[corr_col]:.4f})"
                for idx, row in target_corr.head(10).iterrows()
            ]
        )
    else:
        top_corr_text = "未生成目标相关性结果。"

    id_text = "、".join(map(str, id_cols)) if id_cols else "无"
    target_name = target_col if target_col is not None else "未提供"

    report_lines = [
        "# EDA Report",
        "",
        "## 1. 数据概况",
        "",
        "本报告由通用EDA工具模块自动生成。当前EDA阶段只分析数据结构、变量分布、目标变量分布以及训练集与测试集一致性，不提前确定最终模型。",
        "",
        f"当前任务类型识别为`{task_type}`。",
        "",
        f"训练集规模为{train_rows}行、{train_cols}列。{test_text}",
        "",
        f"识别到的ID字段为：{id_text}。",
        "",
        "## 2. 目标变量分布",
        "",
        f"目标变量为：`{target_name}`。",
        "",
        f"目标变量分布为：{target_text}",
        "",
        "![target_distribution](figures/target_distribution.png)",
        "",
        "如果目标变量存在明显类别不平衡，后续建模阶段不能只使用Accuracy评价模型，应同时关注AUC、F1、Precision、Recall等指标。",
        "",
        "## 3. 字段类型",
        "",
        f"连续数值变量数量为{len(continuous_cols)}，类别型或低基数变量数量为{len(categorical_cols)}。",
        "",
        "字段类型划分结果已保存至`tables/feature_summary.csv`。",
        "",
        "## 4. 缺失值与重复值检查",
        "",
        missing_text,
        "",
        "缺失值结果保存至`tables/missing_summary.csv`。",
        "",
        "重复值结果保存至`tables/duplicate_summary.csv`。",
        "",
        "## 5. 单变量分布",
        "",
        "数值变量分布图保存至`figures/numeric_distributions/`。",
        "",
        "类别变量分布图保存至`figures/categorical_distributions/`。",
        "",
        "该部分主要用于识别偏态分布、异常值、长尾类别和低频类别。",
        "",
        "## 6. 变量与目标变量关系",
        "",
        "数值变量按目标变量分组的统计结果保存至`tables/numeric_by_target_summary.csv`。",
        "",
        "类别变量对应的目标率统计结果保存至`tables/categorical_by_target_summary.csv`。",
        "",
        "相关图像保存至`figures/numeric_by_target/`和`figures/categorical_by_target/`。",
        "",
        "## 7. 相关性分析",
        "",
        f"与目标变量相关性较高的数值变量包括：{top_corr_text}",
        "",
        "相关性矩阵保存至`tables/correlation_matrix.csv`。",
        "",
        "目标变量相关性排序保存至`tables/target_correlation.csv`。",
        "",
        "![target_correlation_top20](figures/target_correlation_top20.png)",
        "",
        "![correlation_heatmap_top_features](figures/correlation_heatmap_top_features.png)",
        "",
        "需要注意，相关性只能反映线性关系，不能直接解释因果关系。低线性相关不代表该变量在非线性模型中无效。",
        "",
        "## 8. 训练集与测试集分布差异",
        "",
        "数值变量的训练集-测试集分布差异保存至`tables/numeric_train_test_shift.csv`。",
        "",
        "类别变量的新水平和缺失水平检查保存至`tables/categorical_train_test_shift.csv`。",
        "",
        "如果测试集出现训练集中没有的新类别，后续建模需要使用能够处理未知类别的编码方式。",
        "",
        "## 9. EDA阶段结论",
        "",
        "本阶段完成了数据质量、字段类型、目标变量分布、单变量分布、变量与目标变量关系、相关性分析以及训练集与测试集分布差异检查。",
        "",
        "后续建模阶段应重点关注：",
        "",
        "1. 目标变量是否存在类别不平衡；",
        "2. 是否存在缺失值、异常值或重复记录；",
        "3. 类别变量是否需要编码；",
        "4. 数值变量是否存在明显偏态或极端值；",
        "5. 测试集是否出现训练集未包含的新类别；",
        "6. 模型评价指标不能只依赖Accuracy。",
        "",
    ]

    report = "\n".join(report_lines)

    report_path = output_dir / "eda_report.md"
    report_path.write_text(report, encoding="utf-8")

    return report_path


def run_basic_tabular_eda(
    train: pd.DataFrame,
    test: pd.DataFrame | None = None,
    target_col: str | None = None,
    output_dir: str | Path = "reports/eda",
    id_cols: Sequence[str] | None = None,
    task_type: TaskType = "auto",
    low_cardinality_threshold: int = 20,
    max_plot_cols: int = 40,
) -> dict[str, object]:
    """
    Run reusable basic EDA for tabular Kaggle-style datasets.
    """
    warnings.filterwarnings("ignore", category=FutureWarning)

    output_dir = ensure_dir(output_dir)
    table_dir = ensure_dir(output_dir / "tables")
    figure_dir = ensure_dir(output_dir / "figures")

    numeric_fig_dir = ensure_dir(figure_dir / "numeric_distributions")
    categorical_fig_dir = ensure_dir(figure_dir / "categorical_distributions")
    numeric_target_fig_dir = ensure_dir(figure_dir / "numeric_by_target")
    categorical_target_fig_dir = ensure_dir(figure_dir / "categorical_by_target")

    if target_col is not None and target_col not in train.columns:
        raise ValueError(f"Target column `{target_col}` is not in train columns.")

    if id_cols is None:
        id_cols = infer_id_cols(train)
    else:
        id_cols = list(id_cols)

    if task_type == "auto":
        detected_task_type = infer_task_type(train, target_col)
    else:
        detected_task_type = task_type

    col_types = split_feature_types(
        train=train,
        target_col=target_col,
        id_cols=id_cols,
        low_cardinality_threshold=low_cardinality_threshold,
    )

    continuous_cols = col_types["continuous_cols"]
    categorical_cols = col_types["categorical_like_cols"]
    numeric_cols = col_types["numeric_cols"]

    train_overview = build_overview(train)
    train_overview.to_csv(table_dir / "train_overview.csv", index=False, encoding="utf-8-sig")

    if test is not None:
        test_overview = build_overview(test)
        test_overview.to_csv(table_dir / "test_overview.csv", index=False, encoding="utf-8-sig")
    else:
        test_overview = pd.DataFrame()

    feature_summary = pd.DataFrame(
        {
            "feature_type": [
                "id_cols",
                "continuous_numeric_cols",
                "categorical_like_cols",
                "raw_categorical_cols",
                "low_card_numeric_cols",
            ],
            "n_features": [
                len(col_types["id_cols"]),
                len(col_types["continuous_cols"]),
                len(col_types["categorical_like_cols"]),
                len(col_types["raw_categorical_cols"]),
                len(col_types["low_card_num_cols"]),
            ],
            "features": [
                ", ".join(map(str, col_types["id_cols"])),
                ", ".join(map(str, col_types["continuous_cols"])),
                ", ".join(map(str, col_types["categorical_like_cols"])),
                ", ".join(map(str, col_types["raw_categorical_cols"])),
                ", ".join(map(str, col_types["low_card_num_cols"])),
            ],
        }
    )
    feature_summary.to_csv(table_dir / "feature_summary.csv", index=False, encoding="utf-8-sig")

    missing_summary = save_missing_summary(train, test, table_dir)
    duplicate_summary = save_duplicate_summary(train, test, id_cols, table_dir)

    if target_col is not None:
        target_dist = save_target_distribution(train, target_col, table_dir, figure_dir)
    else:
        target_dist = None

    numeric_describe = save_numeric_describe(train, continuous_cols, table_dir)

    save_numeric_distribution_plots(
        train=train,
        continuous_cols=continuous_cols,
        output_dir=numeric_fig_dir,
        max_plot_cols=max_plot_cols,
    )

    categorical_count_summary = save_categorical_distribution_plots(
        train=train,
        categorical_cols=categorical_cols,
        table_dir=table_dir,
        output_dir=categorical_fig_dir,
    )

    target_numeric = None
    numeric_by_target_summary = pd.DataFrame()
    categorical_by_target_summary = pd.DataFrame()
    target_corr = pd.DataFrame()

    if target_col is not None and detected_task_type == "binary_classification":
        target_numeric = to_binary_numeric(train[target_col])

        if target_numeric.nunique(dropna=True) == 2:
            numeric_by_target_summary = save_numeric_by_binary_target(
                train=train,
                continuous_cols=continuous_cols,
                target_numeric=target_numeric,
                table_dir=table_dir,
                output_dir=numeric_target_fig_dir,
                max_plot_cols=max_plot_cols,
            )

            categorical_by_target_summary = save_categorical_by_binary_target(
                train=train,
                categorical_cols=categorical_cols,
                target_numeric=target_numeric,
                table_dir=table_dir,
                output_dir=categorical_target_fig_dir,
            )

            target_corr = save_correlation_analysis(
                train=train,
                continuous_cols=continuous_cols,
                target_col=target_col,
                target_numeric=target_numeric,
                table_dir=table_dir,
                figure_dir=figure_dir,
            )

    numeric_shift, categorical_shift = save_train_test_shift(
        train=train,
        test=test,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        id_cols=id_cols,
        table_dir=table_dir,
    )

    report_path = write_eda_report(
        output_dir=output_dir,
        train=train,
        test=test,
        target_col=target_col,
        task_type=detected_task_type,
        id_cols=id_cols,
        continuous_cols=continuous_cols,
        categorical_cols=categorical_cols,
        target_dist=target_dist,
        missing_summary=missing_summary,
        target_corr=target_corr,
    )

    print("EDA finished.")
    print(f"Output directory: {output_dir}")
    print(f"Report path: {report_path}")

    return {
        "task_type": detected_task_type,
        "id_cols": id_cols,
        "continuous_cols": continuous_cols,
        "categorical_cols": categorical_cols,
        "train_overview": train_overview,
        "test_overview": test_overview,
        "feature_summary": feature_summary,
        "missing_summary": missing_summary,
        "duplicate_summary": duplicate_summary,
        "target_distribution": target_dist,
        "numeric_describe": numeric_describe,
        "categorical_count_summary": categorical_count_summary,
        "numeric_by_target_summary": numeric_by_target_summary,
        "categorical_by_target_summary": categorical_by_target_summary,
        "target_correlation": target_corr,
        "numeric_train_test_shift": numeric_shift,
        "categorical_train_test_shift": categorical_shift,
        "report_path": report_path,
    }
