"""
Output formatting utilities for clinical evidence reports.

Provides functions for formatting:
- Therapy comparison tables
- Gene/variant analysis reports
- Multi-search aggregated results
"""

from typing import Dict, List

import pandas as pd


def format_comparison_table(comparison_df: pd.DataFrame) -> str:
    """
    Format therapy comparison DataFrame as a readable table.

    Args:
        comparison_df: DataFrame from compare_therapy_evidence()

    Returns:
        Formatted string table
    """
    if comparison_df.empty:
        return "No therapies to compare."

    output = "📊 **Therapy Comparison Table**\n\n"

    # Header
    output += f"{'Rank':<6} {'Therapy':<30} {'Total':<7} {'Avg':<5} {'Strong':<7} {'Score':<7} {'Diseases':<9}\n"
    output += "=" * 90 + "\n"

    # Rows
    for _, row in comparison_df.iterrows():
        output += (
            f"{row['rank']:<6} "
            f"{row['therapy']:<30} "
            f"{row['total_evidence']:<7} "
            f"{row['avg_rating']:<5.1f} "
            f"{row['strong_evidence']:<7} "
            f"{row['actionability_score']:<7.1f} "
            f"{row['unique_diseases']:<9}\n"
        )

    output += "\n**Legend:**\n"
    output += "- Total: Total evidence count\n"
    output += "- Avg: Average evidence rating (out of 5)\n"
    output += "- Strong: Evidence items with rating > 3\n"
    output += "- Score: Clinical actionability score (0-100)\n"
    output += "- Diseases: Number of unique diseases\n"

    return output


def format_evidence_type_breakdown(comparison_df: pd.DataFrame) -> str:
    """
    Format evidence type breakdown for compared therapies.

    Args:
        comparison_df: DataFrame from compare_therapy_evidence()

    Returns:
        Formatted string with evidence type distributions
    """
    if comparison_df.empty:
        return ""

    output = "\n🔬 **Evidence Type Breakdown**\n\n"

    for _, row in comparison_df.iterrows():
        output += f"**{row['therapy']}:**\n"
        output += f"  - Predictive: {row['predictive_count']}\n"
        output += f"  - Diagnostic: {row['diagnostic_count']}\n"
        output += f"  - Prognostic: {row['prognostic_count']}\n\n"

    return output


def format_recommendation(comparison_df: pd.DataFrame) -> str:
    """
    Format therapy recommendation based on comparison.

    Args:
        comparison_df: DataFrame from compare_therapy_evidence()

    Returns:
        Formatted recommendation string
    """
    if comparison_df.empty:
        return ""

    top_therapy = comparison_df.iloc[0]

    output = "\n💡 **Recommendation**\n\n"

    if top_therapy["actionability_score"] >= 70:
        strength = "strong"
    elif top_therapy["actionability_score"] >= 50:
        strength = "moderate"
    else:
        strength = "weak"

    output += (
        f"Based on evidence analysis, **{top_therapy['therapy']}** has the {strength} evidence base "
        f"(actionability score: {top_therapy['actionability_score']:.1f}/100).\n\n"
    )

    output += "**Rationale:**\n"
    output += f"- {top_therapy['total_evidence']} total evidence items\n"
    output += f"- Average rating: {top_therapy['avg_rating']:.1f}/5\n"
    output += f"- {top_therapy['strong_evidence']} high-quality evidence items\n"
    output += f"- Evidence across {top_therapy['unique_diseases']} disease contexts\n"

    if top_therapy["predictive_count"] > 0:
        output += f"- {top_therapy['predictive_count']} predictive evidence items (clinically actionable)\n"

    return output


def format_gene_variant_overview(
    gene_or_variant: str, df: pd.DataFrame, analysis_type: str
) -> str:
    """
    Format overview section for gene/variant analysis.

    Args:
        gene_or_variant: Name of gene or variant
        df: DataFrame containing evidence
        analysis_type: "gene" or "variant"

    Returns:
        Formatted overview string
    """
    if df.empty:
        return f"🔍 No evidence found for {gene_or_variant}.\n"

    output = f"🧬 **{analysis_type.title()} Analysis: {gene_or_variant}**\n\n"

    output += "📊 **Overview:**\n"
    output += f"- Total Evidence Items: {len(df)}\n"
    output += f"- Average Evidence Rating: {df['evidence_rating'].mean():.2f}/5\n"
    output += f"- Associated Diseases: {df['disease_name'].nunique()}\n"
    output += f"- Associated Therapies: {df['therapy_names'].nunique()}\n"
    output += f"- Evidence Types: {df['evidence_type'].nunique()}\n"

    return output


def format_association_analysis(
    associations_df: pd.DataFrame, group_by: str, top_associations: List[str]
) -> str:
    """
    Format association analysis results.

    Args:
        associations_df: DataFrame from analyze_gene_variant_associations()
        group_by: Grouping field name
        top_associations: List of top association names

    Returns:
        Formatted association analysis string
    """
    if associations_df.empty:
        return ""

    group_name_map = {
        "disease_name": "Disease",
        "therapy_names": "Therapy",
        "evidence_type": "Evidence Type",
    }

    group_label = group_name_map.get(group_by, group_by)

    output = f"\n🔗 **{group_label} Associations** (Top 5):\n\n"

    for idx, row in associations_df.head(5).iterrows():
        try:
            association_name = row.get(group_by, "Unknown") if hasattr(row, 'get') else row[group_by]

            # Skip if association_name is None or NaN
            if association_name is None or (isinstance(association_name, float) and pd.isna(association_name)):
                continue

            evidence_count = int(row.get('evidence_count', 0)) if hasattr(row, 'get') else int(row['evidence_count'])
            avg_rating = float(row.get('avg_rating', 0.0)) if hasattr(row, 'get') else float(row['avg_rating'])
            max_rating = int(row.get('max_rating', 0)) if hasattr(row, 'get') else int(row['max_rating'])
            strength_score = float(row.get('strength_score', 0.0)) if hasattr(row, 'get') else float(row['strength_score'])

            output += (
                f"{idx + 1}. **{association_name}**\n"
                f"   - Evidence Count: {evidence_count}\n"
                f"   - Average Rating: {avg_rating:.2f}/5\n"
                f"   - Highest Rating: {max_rating}/5\n"
                f"   - Strength Score: {strength_score:.1f}\n\n"
            )
        except Exception as e:
            # Skip problematic rows
            continue

    return output


def format_evidence_type_distribution(df: pd.DataFrame) -> str:
    """
    Format evidence type distribution.

    Args:
        df: DataFrame containing evidence items

    Returns:
        Formatted distribution string
    """
    if df.empty:
        return ""

    type_counts = df["evidence_type"].value_counts()

    output = "\n📈 **Evidence Type Distribution:**\n"
    for evidence_type, count in type_counts.items():
        percentage = (count / len(df)) * 100
        output += f"- {evidence_type}: {count} ({percentage:.1f}%)\n"

    return output


def format_research_gaps(gaps: List[str]) -> str:
    """
    Format research gap analysis.

    Args:
        gaps: List of research gap descriptions

    Returns:
        Formatted research gaps string
    """
    if not gaps:
        return ""

    output = "\n🔬 **Research Gaps & Opportunities:**\n\n"

    for gap in gaps:
        output += f"- {gap}\n"

    return output


def format_actionability_score(score: float, df: pd.DataFrame) -> str:
    """
    Format clinical actionability score with interpretation.

    Args:
        score: Actionability score (0-100)
        df: DataFrame used for score calculation

    Returns:
        Formatted actionability assessment
    """
    output = "\n⭐ **Clinical Actionability Assessment:**\n\n"

    # Determine rating
    if score >= 80:
        rating = "VERY HIGH"
        color = "🟢"
    elif score >= 60:
        rating = "HIGH"
        color = "🟢"
    elif score >= 40:
        rating = "MODERATE"
        color = "🟡"
    elif score >= 20:
        rating = "LOW"
        color = "🟠"
    else:
        rating = "VERY LOW"
        color = "🔴"

    output += f"{color} **Actionability Score: {score:.1f}/100 ({rating})**\n\n"

    output += "**Interpretation:**\n"

    if score >= 60:
        output += "This gene/variant has a strong evidence base supporting clinical decision-making. "
        output += "Multiple high-quality studies provide actionable insights.\n"
    elif score >= 40:
        output += "This gene/variant has moderate evidence support. "
        output += "Some clinical guidance available, but additional research would be beneficial.\n"
    else:
        output += "This gene/variant has limited evidence support. "
        output += "Caution advised in clinical decision-making; more research needed.\n"

    # Add contributing factors
    if not df.empty:
        output += "\n**Contributing Factors:**\n"
        output += f"- Evidence volume: {len(df)} items\n"
        output += f"- Average quality: {df['evidence_rating'].mean():.1f}/5\n"
        output += f"- Evidence diversity: {df['evidence_type'].nunique()} types\n"

        if "PREDICTIVE" in df["evidence_type"].values:
            predictive_count = len(df[df["evidence_type"] == "PREDICTIVE"])
            output += f"- Predictive evidence: {predictive_count} items (✓ Clinically actionable)\n"

    return output


def format_top_evidence_entries(df: pd.DataFrame, top_n: int = 5) -> str:
    """
    Format top evidence entries with full details.

    Args:
        df: DataFrame containing evidence items
        top_n: Number of top entries to include

    Returns:
        Formatted top evidence string
    """
    if df.empty:
        return ""

    top_evidences = df.sort_values(by="evidence_rating", ascending=False).head(top_n)

    output = f"\n📌 **Top {top_n} Evidence Entries:**\n\n"

    for idx, (_, row) in enumerate(top_evidences.iterrows(), 1):
        output += (
            f"**{idx}. {row.get('evidence_type', 'N/A')} ({row.get('evidence_direction', 'N/A')})** "
            f"| Rating: {row.get('evidence_rating', 'N/A')}/5\n"
        )
        output += f"   - Disease: {row.get('disease_name', 'N/A')}\n"
        output += f"   - Gene/Variant: {row.get('gene_name', 'N/A')} / {row.get('variant_name', 'N/A')}\n"
        output += f"   - Therapy: {row.get('therapy_names', 'N/A')}\n"
        output += f"   - Description: {row.get('description', 'N/A')[:150]}...\n\n"

    return output


def format_multi_search_summary(results_by_param: Dict[str, pd.DataFrame]) -> str:
    """
    Format summary for multi-parameter search results.

    Args:
        results_by_param: Dictionary mapping search parameters to result DataFrames

    Returns:
        Formatted summary string
    """
    output = "📊 **Multi-Search Summary**\n\n"

    total_results = sum(len(df) for df in results_by_param.values())
    output += f"Total Evidence Items Across All Searches: {total_results}\n\n"

    output += "**Results by Parameter:**\n"
    for param, df in results_by_param.items():
        if not df.empty and "evidence_rating" in df.columns:
            avg_rating = df["evidence_rating"].mean()
            output += f"- **{param}**: {len(df)} items (avg rating: {avg_rating:.2f})\n"
        elif not df.empty:
            output += f"- **{param}**: {len(df)} items (data structure error)\n"
        else:
            output += f"- **{param}**: No results\n"

    return output + "\n"
