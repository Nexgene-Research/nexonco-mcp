"""
Analysis utilities for comparing therapies and analyzing genes/variants.

Provides functions for:
- Therapy comparison and ranking
- Evidence strength scoring
- Gene/variant association analysis
- Clinical actionability scoring
"""

from collections import Counter
from typing import Dict, List, Optional

import pandas as pd


def calculate_evidence_strength(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate evidence strength metrics for a dataframe of evidence items.

    Args:
        df: DataFrame containing evidence items with ratings

    Returns:
        Dictionary with strength metrics:
        - total_count: Total number of evidence items
        - avg_rating: Average evidence rating
        - strong_evidence_count: Count of evidence with rating > 3
        - strong_evidence_pct: Percentage of strong evidence
        - evidence_diversity: Count of unique evidence types
    """
    if df.empty:
        return {
            "total_count": 0,
            "avg_rating": 0.0,
            "strong_evidence_count": 0,
            "strong_evidence_pct": 0.0,
            "evidence_diversity": 0,
        }

    return {
        "total_count": len(df),
        "avg_rating": df["evidence_rating"].mean(),
        "strong_evidence_count": len(df[df["evidence_rating"] > 3]),
        "strong_evidence_pct": (len(df[df["evidence_rating"] > 3]) / len(df)) * 100,
        "evidence_diversity": df["evidence_type"].nunique(),
    }


def calculate_actionability_score(df: pd.DataFrame) -> float:
    """
    Calculate clinical actionability score based on evidence characteristics.

    Score is based on:
    - Evidence count (more is better)
    - Average rating (higher is better)
    - Evidence type diversity (more types = higher confidence)
    - Presence of PREDICTIVE evidence (most actionable)

    Args:
        df: DataFrame containing evidence items

    Returns:
        Actionability score from 0-100
    """
    if df.empty:
        return 0.0

    # Base score from evidence count (logarithmic scale)
    import math

    count_score = min(30, math.log(len(df) + 1) * 10)

    # Rating score (normalized to 30 points max)
    rating_score = (df["evidence_rating"].mean() / 5.0) * 30

    # Diversity score (10 points per unique type, max 20)
    diversity_score = min(20, df["evidence_type"].nunique() * 10)

    # Predictive evidence bonus (20 points if present)
    predictive_score = (
        20 if "PREDICTIVE" in df["evidence_type"].values else 0
    )

    total_score = count_score + rating_score + diversity_score + predictive_score

    return min(100.0, total_score)


def compare_therapy_evidence(
    therapy_dfs: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Compare evidence across multiple therapies.

    Args:
        therapy_dfs: Dictionary mapping therapy names to their evidence DataFrames

    Returns:
        Comparison DataFrame with metrics for each therapy
    """
    comparison_data = []

    for therapy_name, df in therapy_dfs.items():
        if df.empty:
            metrics = {
                "therapy": therapy_name,
                "total_evidence": 0,
                "avg_rating": 0.0,
                "strong_evidence": 0,
                "predictive_count": 0,
                "diagnostic_count": 0,
                "prognostic_count": 0,
                "actionability_score": 0.0,
                "unique_diseases": 0,
                "unique_variants": 0,
            }
        else:
            strength = calculate_evidence_strength(df)
            evidence_types = df["evidence_type"].value_counts().to_dict()

            metrics = {
                "therapy": therapy_name,
                "total_evidence": strength["total_count"],
                "avg_rating": round(strength["avg_rating"], 2),
                "strong_evidence": strength["strong_evidence_count"],
                "predictive_count": evidence_types.get("PREDICTIVE", 0),
                "diagnostic_count": evidence_types.get("DIAGNOSTIC", 0),
                "prognostic_count": evidence_types.get("PROGNOSTIC", 0),
                "actionability_score": round(calculate_actionability_score(df), 1),
                "unique_diseases": df["disease_name"].nunique(),
                "unique_variants": df["variant_name"].nunique(),
            }

        comparison_data.append(metrics)

    comparison_df = pd.DataFrame(comparison_data)

    # Sort by actionability score (descending)
    comparison_df = comparison_df.sort_values(
        "actionability_score", ascending=False
    ).reset_index(drop=True)

    # Add ranking column
    comparison_df.insert(0, "rank", range(1, len(comparison_df) + 1))

    return comparison_df


def analyze_gene_variant_associations(
    df: pd.DataFrame, group_by: str = "disease"
) -> Dict[str, any]:
    """
    Analyze associations for a gene or variant across different contexts.

    Args:
        df: DataFrame containing evidence for a specific gene/variant
        group_by: How to group results - "disease", "therapy", or "evidence_type"

    Returns:
        Dictionary with association analysis:
        - associations: DataFrame grouped by specified field
        - top_associations: List of top 5 associations
        - summary: Overall statistics
    """
    if df.empty:
        return {
            "associations": pd.DataFrame(),
            "top_associations": [],
            "summary": {},
        }

    # Map group_by to actual column name
    group_col_map = {
        "disease": "disease_name",
        "therapy": "therapy_names",
        "evidence_type": "evidence_type",
    }

    if group_by not in group_col_map:
        group_by = "disease"

    group_col = group_col_map[group_by]

    # Check if column exists and has non-null values
    if group_col not in df.columns:
        return {
            "associations": pd.DataFrame(),
            "top_associations": [],
            "summary": {"error": f"Column {group_col} not found in data"},
        }

    # Filter out null values before grouping
    df_clean = df[df[group_col].notna()].copy()

    if df_clean.empty:
        return {
            "associations": pd.DataFrame(),
            "top_associations": [],
            "summary": {"error": f"All values in {group_col} are null"},
        }

    # Group and aggregate
    try:
        grouped = (
            df_clean.groupby(group_col, dropna=True)
            .agg(
                {
                    "id": "count",
                    "evidence_rating": ["mean", "max"],
                }
            )
            .reset_index()
        )

        # Flatten column names
        grouped.columns = [
            group_col,
            "evidence_count",
            "avg_rating",
            "max_rating",
        ]

        # Calculate strength score for each association
        grouped["strength_score"] = (
            grouped["evidence_count"] * 0.5
            + grouped["avg_rating"] * 10
            + grouped["max_rating"] * 5
        )

        # Sort by strength score
        grouped = grouped.sort_values("strength_score", ascending=False).reset_index(
            drop=True
        )

        # Get top 5 associations (convert to string to avoid None issues)
        top_5 = [str(x) for x in grouped.head(5)[group_col].tolist()]

        # Calculate summary
        summary = {
            "total_associations": len(grouped),
            "total_evidence": len(df_clean),
            "avg_evidence_per_association": round(len(df_clean) / len(grouped), 1)
            if len(grouped) > 0
            else 0,
            "strongest_association": str(grouped.iloc[0][group_col])
            if len(grouped) > 0
            else "N/A",
        }
    except Exception as e:
        return {
            "associations": pd.DataFrame(),
            "top_associations": [],
            "summary": {"error": f"Grouping failed: {str(e)}"},
        }

    return {
        "associations": grouped,
        "top_associations": top_5,
        "summary": summary,
    }


def identify_research_gaps(df: pd.DataFrame) -> List[str]:
    """
    Identify potential research gaps based on evidence patterns.

    Args:
        df: DataFrame containing evidence items

    Returns:
        List of research gap descriptions
    """
    gaps = []

    if df.empty:
        return ["No evidence available - complete research gap"]

    # Check for low evidence count
    if len(df) < 5:
        gaps.append(
            f"Limited evidence base (only {len(df)} items) - more research needed"
        )

    # Check for low ratings
    avg_rating = df["evidence_rating"].mean()
    if avg_rating < 2.5:
        gaps.append(
            f"Low average evidence rating ({avg_rating:.1f}/5) - higher quality studies needed"
        )

    # Check for missing evidence types
    evidence_types = set(df["evidence_type"].unique())
    all_types = {"PREDICTIVE", "DIAGNOSTIC", "PROGNOSTIC", "PREDISPOSING"}
    missing_types = all_types - evidence_types

    if missing_types:
        gaps.append(
            f"Missing evidence types: {', '.join(missing_types)} - broader research scope needed"
        )

    # Check for limited disease coverage
    if df["disease_name"].nunique() < 3:
        gaps.append(
            "Limited disease contexts studied - more disease associations should be explored"
        )

    # Check for limited therapy coverage
    if df["therapy_names"].nunique() < 3:
        gaps.append(
            "Few therapies associated - additional therapeutic applications should be investigated"
        )

    return gaps if gaps else ["Well-studied with comprehensive evidence base"]


def get_top_genes_variants(df: pd.DataFrame, top_n: int = 5) -> Dict[str, List[tuple]]:
    """
    Extract top genes and variants from evidence DataFrame.

    Args:
        df: DataFrame containing evidence items
        top_n: Number of top items to return

    Returns:
        Dictionary with top_genes and top_variants as (name, count) tuples
    """
    if df.empty:
        return {"top_genes": [], "top_variants": []}

    gene_counts = Counter(df["gene_name"].dropna())
    variant_counts = Counter(df["variant_name"].dropna())

    return {
        "top_genes": gene_counts.most_common(top_n),
        "top_variants": variant_counts.most_common(top_n),
    }
