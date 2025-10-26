from collections import Counter
from typing import List, Optional

import pandas as pd
import uvicorn
from mcp.server import Server
from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport
from pydantic import Field
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Mount, Route

from .analysis import (
    analyze_gene_variant_associations,
    calculate_actionability_score,
    compare_therapy_evidence,
    identify_research_gaps,
)
from .api import CivicAPIClient
from .formatters import (
    format_actionability_score,
    format_association_analysis,
    format_comparison_table,
    format_evidence_type_breakdown,
    format_evidence_type_distribution,
    format_gene_variant_overview,
    format_multi_search_summary,
    format_recommendation,
    format_research_gaps,
    format_top_evidence_entries,
)

API_VERSION = "0.1.17"
BUILD_TIMESTAMP = "2025-08-12"

mcp = FastMCP(
    name="nexonco",
    description="An advanced MCP Server for accessing and analyzing clinical evidence data, with flexible search options to support precision medicine and oncology research.",
    version=API_VERSION,
)


@mcp.tool(
    name="search_clinical_evidence",
    description=(
        "Perform a flexible search for clinical evidence using combinations of filters such as disease, therapy, "
        "molecular profile, phenotype, evidence type, and direction. This flexible search system allows you to tailor "
        "your query based on the data needed for research or clinical decision-making. It returns a detailed report that "
        "includes summary statistics, a top 10 evidence listing, citation sources, and a disclaimer."
    ),
)
def search_clinical_evidence(
    disease_name: Optional[str] = Field(
        default="",
        description="Name of the disease to filter evidence by (e.g., 'Von Hippel-Lindau Disease', 'Lung Non-small Cell Carcinoma', 'Colorectal Cancer', 'Chronic Myeloid Leukemia', 'Glioblastoma'..). Case-insensitive and optional.",
    ),
    therapy_name: Optional[str] = Field(
        default="",
        description="Therapy or drug name involved in the evidence (e.g., 'Cetuximab', 'Imatinib', 'trastuzumab', 'Lapatinib'..). Optional.",
    ),
    molecular_profile_name: Optional[str] = Field(
        default="",
        description="Molecular profile or gene name or variant name (e.g., 'EGFR L858R', 'BRAF V600E', 'KRAS', 'PIK3CA'..). Optional.",
    ),
    phenotype_name: Optional[str] = Field(
        default="",
        description="Name of the phenotype or histological subtype (e.g., 'Hemangioblastoma', 'Renal cell carcinoma', 'Retinal capillary hemangioma', 'Pancreatic cysts', 'Childhood onset'..). Optional.",
    ),
    evidence_type: Optional[str] = Field(
        default="",
        description="Evidence classification: 'PREDICTIVE', 'DIAGNOSTIC', 'PROGNOSTIC', 'PREDISPOSING', or 'FUNCTIONAL'. Optional.",
    ),
    evidence_direction: Optional[str] = Field(
        default="",
        description="Direction of the evidence: 'SUPPORTS' or 'DOES_NOT_SUPPORT'. Indicates if the evidence favors the association.",
    ),
    filter_strong_evidence: bool = Field(
        default=False,
        description="If set to True, only evidence with a rating above 3 will be included, indicating high-confidence evidence. However, the number of returned evidence items may be quite low.",
    ),
) -> str:
    """
    Query clinical evidence records using flexible combinations of disease, therapy, molecular profile,
    phenotype, and other evidence characteristics. Returns a formatted report containing a summary of findings,
    most common genes and therapies, and highlights of top-ranked evidence entries including source URLs and citations.

    This tool is designed to streamline evidence exploration in precision oncology by adapting to various research
    or clinical inquiry contexts.

    Returns:
        str: A human-readable report summarizing relevant evidence, key statistics, and literature references.
    """

    client = CivicAPIClient()

    disease_name = None if disease_name == "" else disease_name
    therapy_name = None if therapy_name == "" else therapy_name
    molecular_profile_name = (
        None if molecular_profile_name == "" else molecular_profile_name
    )
    phenotype_name = None if phenotype_name == "" else phenotype_name
    evidence_type = None if evidence_type == "" else evidence_type
    evidence_direction = None if evidence_direction == "" else evidence_direction

    df: pd.DataFrame = client.search_evidence(
        disease_name=disease_name,
        therapy_name=therapy_name,
        molecular_profile_name=molecular_profile_name,
        phenotype_name=phenotype_name,
        evidence_type=evidence_type,
        evidence_direction=evidence_direction,
        filter_strong_evidence=filter_strong_evidence,
    )

    if df.empty:
        return "🔍 No evidence found for the specified filters."

    # ---------------------------------
    # 1. Summary Statistics Section
    # ---------------------------------
    total_items = len(df)
    avg_rating = df["evidence_rating"].mean()

    # Frequency counters for each key attribute
    disease_counter = Counter(df["disease_name"].dropna())
    gene_counter = Counter(df["gene_name"].dropna())
    variant_counter = Counter(df["variant_name"].dropna())
    therapy_counter = Counter(df["therapy_names"].dropna())
    phenotype_counter = Counter(df["phenotype_name"].dropna())

    # Prepare top-3 summary for each attribute
    def format_top(counter: Counter) -> str:
        return (
            ", ".join(f"{item} ({count})" for item, count in counter.most_common(3))
            if counter
            else "N/A"
        )

    top_diseases = format_top(disease_counter)
    top_genes = format_top(gene_counter)
    top_variants = format_top(variant_counter)
    top_therapies = format_top(therapy_counter)
    top_phenotypes = format_top(phenotype_counter)

    stats_section = (
        f"📊 **Summary Statistics**\n"
        f"- Total Evidence Items: {total_items}\n"
        f"- Average Evidence Rating: {avg_rating:.2f}\n"
        f"- Top Diseases: {top_diseases}\n"
        f"- Top Genes: {top_genes}\n"
        f"- Top Variants: {top_variants}\n"
        f"- Top Therapies: {top_therapies}\n"
        f"- Top Phenotypes: {top_phenotypes}\n"
    )

    # ---------------------------------
    # 2. Top 10 Evidence Listing Section
    # ---------------------------------
    top_evidences = df.sort_values(by="evidence_rating", ascending=False).head(10)
    evidence_section = "📌 **Top 10 Evidence Entries**\n"
    for _, row in top_evidences.iterrows():
        evidence_section += (
            f"\n🔹 **{row.get('evidence_type', 'N/A')} ({row.get('evidence_direction', 'N/A')})** | Rating: {row.get('evidence_rating', 'N/A')}\n"
            f"- Disease: {row.get('disease_name', 'N/A')}\n"
            f"- Phenotype: {row.get('phenotype_name', 'N/A')}\n"
            f"- Gene/Variant: {row.get('gene_name', 'N/A')} / {row.get('variant_name', 'N/A')}\n"
            f"- Therapy: {row.get('therapy_names', 'N/A')}\n"
            f"- Description: {row.get('description', 'N/A')}\n"
        )

    # ---------------------------------
    # 3. Sources & Citations Section
    # ---------------------------------
    citation_section = "🔗 **Sources & Citations**\n"
    sources = client.get_sources(top_evidences["id"].tolist())
    for _, row in pd.DataFrame(sources).iterrows():
        citation_section += (
            f"• {row.get('citation', 'N/A')} - {row.get('sourceUrl', 'N/A')}\n"
        )

    # ---------------------------------
    # 4. Disclaimer Section
    # ---------------------------------
    disclaimer = "\n⚠️ **Disclaimer:** This tool is intended exclusively for research purposes. It is not a substitute for professional medical advice, diagnosis, or treatment."

    # ---------------------------------
    # Combine All Sections into Final Report
    # ---------------------------------
    final_report = (
        f"{stats_section}\n"
        f"{evidence_section}\n"
        f"{citation_section}\n"
        f"{disclaimer}"
    )

    return final_report


@mcp.tool(
    name="multi_search_clinical_evidence",
    description=(
        "Perform multi-parameter searches for clinical evidence across multiple diseases, therapies, "
        "or molecular profiles simultaneously. This tool enables batch querying to compare evidence "
        "across different contexts. Results are aggregated and presented with cross-parameter analysis."
    ),
)
def multi_search_clinical_evidence(
    disease_names: Optional[List[str]] = Field(
        default=None,
        description="List of disease names to search for (e.g., ['Lung Cancer', 'Colorectal Cancer']). Optional.",
    ),
    therapy_names: Optional[List[str]] = Field(
        default=None,
        description="List of therapy names to search for (e.g., ['Cetuximab', 'Panitumumab']). Optional.",
    ),
    molecular_profile_names: Optional[List[str]] = Field(
        default=None,
        description="List of molecular profiles/genes/variants to search for (e.g., ['KRAS', 'EGFR', 'BRAF V600E']). Optional.",
    ),
    evidence_type: Optional[str] = Field(
        default="",
        description="Evidence classification to filter all searches: 'PREDICTIVE', 'DIAGNOSTIC', 'PROGNOSTIC', 'PREDISPOSING', or 'FUNCTIONAL'. Optional.",
    ),
    filter_strong_evidence: bool = Field(
        default=False,
        description="If True, only include evidence with rating > 3 across all searches.",
    ),
) -> str:
    """
    Perform batch searches across multiple diseases, therapies, or molecular profiles.
    Returns aggregated results with summary statistics for each search parameter.

    This tool is useful for:
    - Comparing evidence across multiple diseases
    - Analyzing multiple gene variants simultaneously
    - Exploring therapeutic options for related conditions

    Returns:
        str: Formatted report with results grouped by search parameter, including
        aggregate statistics and cross-parameter insights.
    """
    client = CivicAPIClient()

    # Convert empty strings to None
    evidence_type = None if evidence_type == "" else evidence_type

    # Ensure at least one parameter list is provided
    if not any([disease_names, therapy_names, molecular_profile_names]):
        return "❌ Error: Please provide at least one list of diseases, therapies, or molecular profiles to search."

    # Perform batch search
    results_by_param = client.search_evidence_batch(
        disease_names=disease_names,
        therapy_names=therapy_names,
        molecular_profile_names=molecular_profile_names,
        evidence_type=evidence_type,
        filter_strong_evidence=filter_strong_evidence,
    )

    if not results_by_param:
        return "🔍 No search parameters provided or no results found."

    # Check if all results are empty
    if all(df.empty for df in results_by_param.values()):
        return "🔍 No evidence found for any of the specified parameters."

    # Generate summary
    report = format_multi_search_summary(results_by_param)

    # Add detailed results for each parameter
    report += "---\n\n## **Detailed Results by Parameter**\n\n"

    for param_name, df in results_by_param.items():
        if df.empty:
            report += f"### {param_name}\n\nNo evidence found.\n\n"
            continue

        # Check if DataFrame has required columns
        if "evidence_rating" not in df.columns:
            report += f"### {param_name}\n\nData structure error - missing evidence rating.\n\n"
            continue

        # Statistics for this parameter
        total_items = len(df)
        avg_rating = df["evidence_rating"].mean()

        report += f"### {param_name}\n\n"
        report += f"**Statistics:**\n"
        report += f"- Total Evidence: {total_items}\n"
        report += f"- Average Rating: {avg_rating:.2f}/5\n"
        report += f"- Strong Evidence: {len(df[df['evidence_rating'] > 3])}\n"

        # Top diseases, genes, therapies for this parameter
        if "disease_name" in df.columns:
            top_diseases = df["disease_name"].value_counts().head(3)
            report += f"- Top Diseases: {', '.join(f'{d} ({c})' for d, c in top_diseases.items())}\n"

        if "gene_name" in df.columns:
            top_genes = df["gene_name"].value_counts().head(3)
            report += f"- Top Genes: {', '.join(f'{g} ({c})' for g, c in top_genes.items())}\n"

        # Top 3 evidence entries
        report += "\n**Top 3 Evidence Entries:**\n\n"
        top_3 = df.sort_values(by="evidence_rating", ascending=False).head(3)

        for idx, (_, row) in enumerate(top_3.iterrows(), 1):
            report += (
                f"{idx}. **{row.get('evidence_type', 'N/A')}** | "
                f"Rating: {row.get('evidence_rating', 'N/A')}/5\n"
                f"   - Disease: {row.get('disease_name', 'N/A')}\n"
                f"   - Gene/Variant: {row.get('gene_name', 'N/A')} / {row.get('variant_name', 'N/A')}\n"
                f"   - Therapy: {row.get('therapy_names', 'N/A')}\n\n"
            )

        report += "---\n\n"

    # Add cross-parameter analysis
    all_dfs = [df for df in results_by_param.values() if not df.empty]
    if len(all_dfs) > 1:
        try:
            combined_df = pd.concat(all_dfs, ignore_index=True)

            report += "## **Cross-Parameter Analysis**\n\n"

            # Find genes/variants that appear across multiple search parameters
            if "gene_name" in combined_df.columns:
                # Filter out null gene names before grouping
                combined_clean = combined_df[combined_df["gene_name"].notna()].copy()

                if not combined_clean.empty and "evidence_rating" in combined_clean.columns:
                    gene_counts = combined_clean.groupby("gene_name", dropna=True)["evidence_rating"].agg(
                        ["count", "mean"]
                    )
                    gene_counts = gene_counts.sort_values("count", ascending=False).head(5)

                    if not gene_counts.empty:
                        report += "**Most Frequent Genes Across All Searches:**\n"
                        for gene, row in gene_counts.iterrows():
                            # Safely access row values
                            if gene and not pd.isna(gene):
                                count_val = row.get("count", 0) if hasattr(row, 'get') else row[0]
                                mean_val = row.get("mean", 0.0) if hasattr(row, 'get') else row[1]
                                report += f"- {gene}: {int(count_val)} occurrences (avg rating: {mean_val:.2f})\n"

            report += "\n"
        except Exception as e:
            report += f"\n⚠️ Cross-parameter analysis could not be completed: {str(e)}\n\n"

    # Disclaimer
    report += "\n⚠️ **Disclaimer:** This tool is intended exclusively for research purposes. It is not a substitute for professional medical advice, diagnosis, or treatment."

    return report


@mcp.tool(
    name="compare_therapies",
    description=(
        "Compare multiple therapies based on clinical evidence for a specific disease or molecular profile context. "
        "Provides side-by-side comparison of evidence strength, quality, and actionability. "
        "Returns ranked recommendations based on evidence analysis."
    ),
)
def compare_therapies(
    therapy_names: List[str] = Field(
        ...,
        description="List of therapy names to compare (e.g., ['Cetuximab', 'Panitumumab', 'Bevacizumab']). Required.",
    ),
    disease_name: Optional[str] = Field(
        default="",
        description="Disease context for the comparison (e.g., 'Colorectal Cancer'). Highly recommended for meaningful comparison. Optional.",
    ),
    molecular_profile_name: Optional[str] = Field(
        default="",
        description="Molecular profile context (e.g., 'KRAS', 'EGFR L858R'). Narrows comparison to specific genetic contexts. Optional.",
    ),
    evidence_type: Optional[str] = Field(
        default="",
        description="Filter by evidence type: 'PREDICTIVE', 'DIAGNOSTIC', 'PROGNOSTIC', 'PREDISPOSING', or 'FUNCTIONAL'. Optional.",
    ),
    filter_strong_evidence: bool = Field(
        default=False,
        description="If True, only include high-quality evidence (rating > 3) in comparison.",
    ),
) -> str:
    """
    Compare multiple therapies to identify the most evidence-supported option for a clinical context.

    This tool is useful for:
    - Treatment decision support
    - Identifying therapies with strongest evidence base
    - Comparing therapeutic options for specific mutations

    Returns:
        str: Comprehensive comparison report with rankings, metrics, and recommendations.
    """
    client = CivicAPIClient()

    # Input validation
    if not therapy_names or len(therapy_names) < 2:
        return "❌ Error: Please provide at least 2 therapies to compare."

    # Convert empty strings to None
    disease_name = None if disease_name == "" else disease_name
    molecular_profile_name = (
        None if molecular_profile_name == "" else molecular_profile_name
    )
    evidence_type = None if evidence_type == "" else evidence_type

    # Retrieve evidence for each therapy
    therapy_data = client.compare_therapies_data(
        therapy_names=therapy_names,
        disease_name=disease_name,
        molecular_profile_name=molecular_profile_name,
        evidence_type=evidence_type,
        filter_strong_evidence=filter_strong_evidence,
    )

    # Check if all therapies have no evidence
    if all(df.empty for df in therapy_data.values()):
        return "🔍 No evidence found for any of the specified therapies in the given context."

    # Perform comparison analysis
    comparison_df = compare_therapy_evidence(therapy_data)

    # Build report
    report = "# 🏥 **Therapy Comparison Report**\n\n"

    # Add context information
    if disease_name or molecular_profile_name:
        report += "**Comparison Context:**\n"
        if disease_name:
            report += f"- Disease: {disease_name}\n"
        if molecular_profile_name:
            report += f"- Molecular Profile: {molecular_profile_name}\n"
        if evidence_type:
            report += f"- Evidence Type: {evidence_type}\n"
        report += "\n"

    # Add comparison table
    report += format_comparison_table(comparison_df)

    # Add evidence type breakdown
    report += format_evidence_type_breakdown(comparison_df)

    # Add recommendation
    report += format_recommendation(comparison_df)

    # Add detailed evidence for top therapy
    top_therapy = comparison_df.iloc[0]["therapy"]
    top_therapy_df = therapy_data[top_therapy]

    if not top_therapy_df.empty:
        report += f"\n## **Detailed Evidence for Top-Ranked Therapy: {top_therapy}**\n\n"
        report += format_top_evidence_entries(top_therapy_df, top_n=5)

    # Add disclaimer
    report += "\n⚠️ **Disclaimer:** This comparison is based on available clinical evidence and is intended for research purposes only. It is not a substitute for professional medical judgment or clinical guidelines."

    return report


@mcp.tool(
    name="analyze_gene_variant",
    description=(
        "Perform comprehensive evidence-based analysis for a specific gene or variant. "
        "Analyzes disease associations, therapy connections, evidence strength, and clinical actionability. "
        "Identifies research gaps and provides actionability scoring to guide clinical decision-making."
    ),
)
def analyze_gene_variant(
    gene_or_variant_name: str = Field(
        ...,
        description="Name of the gene or variant to analyze (e.g., 'BRAF', 'EGFR L858R', 'KRAS G12C'). Required.",
    ),
    analysis_type: str = Field(
        default="auto",
        description="Type of analysis: 'gene' for gene-level analysis, 'variant' for variant-specific analysis, 'auto' to detect automatically. Default: 'auto'.",
    ),
    group_by: str = Field(
        default="disease",
        description="How to organize association analysis: 'disease' (default), 'therapy', or 'evidence_type'.",
    ),
    disease_context: Optional[str] = Field(
        default="",
        description="Optional disease context to narrow analysis (e.g., 'Lung Cancer'). If provided, analysis focuses on this disease.",
    ),
    filter_strong_evidence: bool = Field(
        default=False,
        description="If True, only include high-quality evidence (rating > 3) in analysis.",
    ),
) -> str:
    """
    Comprehensive analysis of a gene or variant including:
    - Evidence volume and quality metrics
    - Disease associations with strength scoring
    - Therapy associations and effectiveness
    - Evidence type distribution
    - Clinical actionability assessment
    - Research gap identification

    This tool is useful for:
    - Understanding clinical significance of variants
    - Identifying therapeutic opportunities
    - Assessing evidence maturity for biomarkers

    Returns:
        str: Detailed analysis report with multiple sections covering all aspects of the gene/variant.
    """
    client = CivicAPIClient()

    # Convert empty strings to None
    disease_context = None if disease_context == "" else disease_context

    # Determine analysis type if auto
    if analysis_type == "auto":
        # Simple heuristic: if name contains spaces or special characters, likely a variant
        if " " in gene_or_variant_name or any(
            c.isdigit() for c in gene_or_variant_name
        ):
            analysis_type = "variant"
        else:
            analysis_type = "gene"

    # Retrieve evidence data
    df = client.analyze_molecular_profile_data(
        molecular_profile_name=gene_or_variant_name,
        disease_name=disease_context,
        filter_strong_evidence=filter_strong_evidence,
    )

    if df.empty:
        return f"🔍 No evidence found for {gene_or_variant_name}."

    # Generate report
    report = format_gene_variant_overview(gene_or_variant_name, df, analysis_type)

    # Calculate actionability score
    actionability = calculate_actionability_score(df)
    report += format_actionability_score(actionability, df)

    # Evidence type distribution
    report += format_evidence_type_distribution(df)

    # Association analysis
    associations = analyze_gene_variant_associations(df, group_by=group_by)

    if not associations["associations"].empty:
        report += format_association_analysis(
            associations["associations"],
            group_by + "_name" if group_by != "evidence_type" else "evidence_type",
            associations["top_associations"],
        )

    # Research gaps
    gaps = identify_research_gaps(df)
    report += format_research_gaps(gaps)

    # Top evidence entries
    report += format_top_evidence_entries(df, top_n=10)

    # Get citations for top evidence
    top_evidences = df.sort_values(by="evidence_rating", ascending=False).head(10)
    if not top_evidences.empty:
        sources = client.get_sources(top_evidences["id"].tolist())
        report += "\n🔗 **Sources & Citations** (Top 10 Evidence):\n\n"
        for source in sources:
            if isinstance(source, dict):
                report += f"• {source.get('citation', 'N/A')} - {source.get('sourceUrl', 'N/A')}\n"

    # Summary insights
    report += "\n## **Summary Insights**\n\n"

    if actionability >= 70:
        report += f"✅ **{gene_or_variant_name}** is well-characterized with strong clinical evidence. "
        report += "This biomarker has robust evidence supporting clinical decision-making.\n\n"
    elif actionability >= 40:
        report += f"⚠️ **{gene_or_variant_name}** has moderate evidence support. "
        report += "Some clinical guidance available, but additional research would strengthen confidence.\n\n"
    else:
        report += f"❌ **{gene_or_variant_name}** has limited evidence support. "
        report += "Caution advised in clinical applications; more research needed.\n\n"

    # Key metrics
    report += "**Key Metrics:**\n"
    report += f"- Evidence Items: {len(df)}\n"
    report += f"- Average Quality: {df['evidence_rating'].mean():.2f}/5\n"
    report += f"- Disease Contexts: {df['disease_name'].nunique()}\n"
    report += f"- Therapeutic Options: {df['therapy_names'].nunique()}\n"
    report += f"- Actionability Score: {actionability:.1f}/100\n"

    # Disclaimer
    report += "\n⚠️ **Disclaimer:** This analysis is based on available clinical evidence and is intended for research purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment."

    return report


async def healthcheck(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok", "message": "Server is healthy."})


async def version(request):
    return JSONResponse({"version": API_VERSION, "build": BUILD_TIMESTAMP})


async def homepage(request: Request) -> HTMLResponse:
    html_content = """
    <!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Nexonco MCP Server</title>
    <style>
        :root {
            --primary: #0f172a;
            --secondary: #0369a1;
            --accent: #0284c7;
            --light-blue: #e0f2fe;
            --light: #ffffff;
            --dark: #0f172a;
            --gray: #64748b;
            --light-gray: #f1f5f9;
            --border-radius: 4px;
            --shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
            color: var(--dark);
            background-color: #f8fafc;
            line-height: 1.6;
        }

        .container {
            max-width: 1100px;
            margin: 0 auto;
            padding: 0 20px;
        }

        header {
            background-color: #000;
            color: white;
            padding: 0;
        }

        .banner-container {
            width: 100%;
            max-width: 100%;
            margin: 0 auto;
        }

        .banner {
            width: 100%;
            max-width: 100%;
            height: auto;
            display: block;
        }

        .subtitle {
            text-align: center;
            padding: 0.75rem 0;
            background-color: #000;
            color: white;
            font-size: 0.95rem;
            letter-spacing: 0.5px;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
        }

        .main-content {
            padding: 2.5rem 0;
        }

        .status-bar {
            display: flex;
            align-items: center;
            background-color: white;
            padding: 0.75rem 1.5rem;
            border-radius: var(--border-radius);
            margin-bottom: 2rem;
            box-shadow: var(--shadow);
        }

        .status-indicator {
            display: inline-flex;
            align-items: center;
            padding: 0.4rem 0.8rem;
            background-color: rgba(34, 197, 94, 0.1);
            border-radius: 2rem;
            color: #15803d;
            font-weight: 500;
            font-size: 0.9rem;
            margin-right: auto;
        }

        .status-indicator::before {
            content: "";
            display: inline-block;
            width: 8px;
            height: 8px;
            background-color: #22c55e;
            border-radius: 50%;
            margin-right: 8px;
        }

        .modern-layout {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 1.5rem;
        }

        .connection-section {
            grid-column: 1 / -1;
            margin-bottom: 1.5rem;
        }

        .full-width {
            grid-column: 1 / -1;
        }

        h2 {
            font-size: 1.25rem;
            margin: 0 0 1.25rem;
            color: var(--primary);
            font-weight: 600;
            letter-spacing: 0.5px;
            text-transform: uppercase;
            display: flex;
            align-items: center;
        }

        h2 svg {
            margin-right: 0.5rem;
            width: 20px;
            height: 20px;
        }

        p {
            margin-bottom: 1rem;
            color: #334155;
            font-size: 0.95rem;
        }

        strong {
            color: var(--primary);
            font-weight: 600;
        }

        .card {
            background-color: white;
            border-radius: var(--border-radius);
            padding: 1.75rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        }

        .connection-card {
            background-color: var(--light-blue);
            border-left: 4px solid var(--accent);
        }

        .about-card {
            height: 100%;
        }

        .api-card {
            background-color: #f8fafc;
            height: 100%;
            border-left: 4px solid #6366f1;
        }

        .query-list {
            list-style: none;
            margin-top: 1rem;
        }

        .query-item {
            padding: 0.85rem 1rem;
            margin-bottom: 0.75rem;
            background-color: #f1f5f9;
            border-radius: var(--border-radius);
            border-left: 3px solid var(--secondary);
            font-family: monospace;
            font-size: 0.9rem;
            color: #334155;
        }

        .button {
            background-color: var(--secondary);
            color: white;
            border: none;
            padding: 0.7rem 1.4rem;
            margin: 1rem 0.5rem 1rem 0;
            cursor: pointer;
            border-radius: var(--border-radius);
            font-weight: 500;
            transition: all 0.2s ease;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-size: 0.85rem;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
        }

        .button:hover {
            background-color: var(--accent);
            transform: translateY(-1px);
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        }

        .button.secondary {
            background-color: #e2e8f0;
            color: #334155;
        }

        .button.secondary:hover {
            background-color: #cbd5e1;
        }

        .status {
            border: 1px solid #e2e8f0;
            padding: 1rem;
            min-height: 20px;
            margin-top: 1rem;
            border-radius: var(--border-radius);
            color: #64748b;
            background-color: #f8fafc;
            font-family: monospace;
        }

        .status.connected {
            border-color: rgba(34, 197, 94, 0.3);
            color: #15803d;
            background-color: rgba(34, 197, 94, 0.05);
        }

        .status.error {
            border-color: rgba(239, 68, 68, 0.3);
            color: #b91c1c;
            background-color: rgba(239, 68, 68, 0.05);
        }

        code {
            font-family: monospace;
            background-color: rgba(99, 102, 241, 0.1);
            padding: 0.2rem 0.4rem;
            border-radius: 2px;
            font-size: 0.9rem;
            color: #4f46e5;
        }

        .disclaimer {
            background-color: rgba(239, 68, 68, 0.05);
            border-left: 3px solid #ef4444;
            padding: 1rem;
            margin: 1.25rem 0 0;
            border-radius: var(--border-radius);
            font-size: 0.9rem;
            color: #7f1d1d;
        }

        .api-details {
            display: flex;
            align-items: flex-start;
            margin-top: 0.5rem;
        }

        .api-icon {
            flex-shrink: 0;
            width: 40px;
            height: 40px;
            background-color: rgba(99, 102, 241, 0.1);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-right: 1rem;
            color: #4f46e5;
        }

        .api-text {
            flex-grow: 1;
        }

        .links {
            display: flex;
            gap: 1rem;
            margin-top: 1rem;
            flex-wrap: wrap;
        }

        .links a {
            color: var(--secondary);
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            transition: all 0.2s;
            font-size: 0.9rem;
            padding: 0.5rem 0.75rem;
            background-color: #f1f5f9;
            border-radius: var(--border-radius);
            border: 1px solid rgba(0, 0, 0, 0.05);
        }

        .links a:hover {
            background-color: #e2e8f0;
            transform: translateY(-1px);
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }

        .icon {
            width: 16px;
            height: 16px;
            margin-right: 8px;
        }

        footer {
            color: #64748b;
            padding: 1.5rem 0;
            margin-top: 2rem;
            text-align: center;
            font-size: 0.85rem;
            border-top: 1px solid #e2e8f0;
            background-color: white;
        }

        footer a {
            color: var(--secondary);
            text-decoration: none;
        }

        footer a:hover {
            text-decoration: underline;
        }

        @media (max-width: 768px) {
            .modern-layout {
                grid-template-columns: 1fr;
            }
            
            .links {
                gap: 0.75rem;
            }
            
            .links a {
                padding: 0.4rem 0.6rem;
                font-size: 0.85rem;
            }
        }
    </style>
</head>
<body>
    <header>
        <div class="banner-container">
            <img src="https://github.com/user-attachments/assets/c2ec59e8-ff8c-40e1-b66d-17998fe67ecf" alt="Nexonco MCP Banner" class="banner">
        </div>
        <div class="subtitle">
            <div class="container">
                Clinical Evidence Data Analysis for Precision Oncology
            </div>
            <br>
            <img src="http://nanda-registry.com/api/v1/verification/badge/c6284608-6bce-4417-a170-da6c1a117616" alt="Verified NANDA MCP Server" />
            <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="MIT License" />
        </div>
    </header>

    <main class="main-content">
        <div class="container">
            <!-- Status Bar -->
            <div class="status-bar">
                <div class="status-indicator">Server is running correctly</div>
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <rect x="2" y="2" width="20" height="8" rx="2" ry="2"></rect>
                    <rect x="2" y="14" width="20" height="8" rx="2" ry="2"></rect>
                    <line x1="6" y1="6" x2="6.01" y2="6"></line>
                    <line x1="6" y1="18" x2="6.01" y2="18"></line>
                </svg>
            </div>

            <!-- Connection Section (Full Width) -->
            <div class="connection-section">
                <div class="card connection-card">
                    <h2>
                        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M5 12.55a11 11 0 0 1 14.08 0"></path>
                            <path d="M1.42 9a16 16 0 0 1 21.16 0"></path>
                            <path d="M8.53 16.11a6 6 0 0 1 6.95 0"></path>
                            <line x1="12" y1="20" x2="12.01" y2="20"></line>
                        </svg>
                        Server Connection
                    </h2>
                    <p>Test your connection to the MCP server:</p>
                    
                    <button id="connect-button" class="button">Connect to SSE</button>
                    <div id="disconnect-container"></div>
                    
                    <div class="status" id="status">Connection status will appear here...</div>
                </div>
            </div>

            <!-- Modern Asymmetric Layout -->
            <div class="modern-layout">
                <!-- About Card (Left Column - Wider) -->
                <div class="card about-card">
                    <h2>
                        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <circle cx="12" cy="12" r="10"></circle>
                            <line x1="12" y1="16" x2="12" y2="12"></line>
                            <line x1="12" y1="8" x2="12.01" y2="8"></line>
                        </svg>
                        About Nexonco
                    </h2>
                    <p><strong>Nexonco</strong> is an advanced MCP Server for accessing and analyzing clinical evidence data, with flexible search across variants, diseases, drugs, and phenotypes to support precision oncology research.</p>
                    
                    <div class="disclaimer">
                        <strong>⚠️ Disclaimer:</strong> This tool is intended exclusively for research purposes. It is not a substitute for professional medical advice, diagnosis, or treatment.
                    </div>
                </div>

                <!-- API Card (Right Column - Narrower) -->
                <div class="card api-card">
                    <h2>
                        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <polyline points="16 18 22 12 16 6"></polyline>
                            <polyline points="8 6 2 12 8 18"></polyline>
                        </svg>
                        API Information
                    </h2>
                    
                    <div class="api-details">
                        <div class="api-icon">
                            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                                <polyline points="14 2 14 8 20 8"></polyline>
                                <line x1="16" y1="13" x2="8" y2="13"></line>
                                <line x1="16" y1="17" x2="8" y2="17"></line>
                                <polyline points="10 9 9 9 8 9"></polyline>
                            </svg>
                        </div>
                        <div class="api-text">
                            <p>The server provides the <code>search_clinical_evidence</code> tool for querying clinical evidence data with filters for disease, therapy, molecular profile, phenotype, and evidence type.</p>
                        </div>
                    </div>
                </div>

                <!-- Sample Queries Card (Full Width) -->
                <div class="card full-width">
                    <h2>
                        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <line x1="22" y1="2" x2="11" y2="13"></line>
                            <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
                        </svg>
                        Sample Queries
                    </h2>
                    <ul class="query-list">
                        <li class="query-item">Find predictive evidence for colorectal cancer therapies involving KRAS mutations.</li>
                        <li class="query-item">Are there studies on Imatinib for leukemia?</li>
                        <li class="query-item">What therapies are linked to pancreatic cancer evidence?</li>
                    </ul>
                </div>

                <!-- Links Card (Full Width) -->
                <div class="card full-width">
                    <h2>
                        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"></path>
                            <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"></path>
                        </svg>
                        Links
                    </h2>
                    <div class="links">
                        <a href="https://www.nexgene.ai" target="_blank">
                            <svg class="icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                <circle cx="12" cy="12" r="10"></circle>
                                <line x1="2" y1="12" x2="22" y2="12"></line>
                                <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"></path>
                            </svg>
                            Nexgene AI
                        </a>
                        <a href="https://www.linkedin.com/company/nexgene" target="_blank">
                            <svg class="icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                <path d="M16 8a6 6 0 0 1 6 6v7h-4v-7a2 2 0 0 0-2-2 2 2 0 0 0-2 2v7h-4v-7a6 6 0 0 1 6-6z"></path>
                                <rect x="2" y="9" width="4" height="12"></rect>
                                <circle cx="4" cy="4" r="2"></circle>
                            </svg>
                            LinkedIn
                        </a>
                        <a href="https://github.com/Nexgene-Research/nexonco-mcp" target="_blank">
                            <svg class="icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                <path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"></path>
                            </svg>
                            GitHub
                        </a>
                    </div>
                </div>
            </div>
        </div>
    </main>

    <footer>
        <div class="container">
            <p>Nexonco MCP Server &copy; 2025 <a href="https://www.nexgene.ai" target="_blank">Nexgene AI</a> | MIT License</p>
        </div>
    </footer>

    <script>
        // Server connection functionality
        document.getElementById('connect-button').addEventListener('click', function() {
            const statusDiv = document.getElementById('status');
            const disconnectContainer = document.getElementById('disconnect-container');
            
            try {
                const eventSource = new EventSource('/sse');
                
                statusDiv.textContent = 'Connecting...';
                statusDiv.className = 'status';
                
                eventSource.onopen = function() {
                    statusDiv.textContent = 'Connected to SSE';
                    statusDiv.className = 'status connected';
                };
                
                eventSource.onerror = function() {
                    statusDiv.textContent = 'Error connecting to SSE';
                    statusDiv.className = 'status error';
                    eventSource.close();
                };
                
                eventSource.onmessage = function(event) {
                    statusDiv.textContent = 'Received: ' + event.data;
                    statusDiv.className = 'status connected';
                };
                
                // Add a disconnect option
                disconnectContainer.innerHTML = '';
                const disconnectButton = document.createElement('button');
                disconnectButton.textContent = 'Disconnect';
                disconnectButton.className = 'button secondary';
                disconnectButton.addEventListener('click', function() {
                    eventSource.close();
                    statusDiv.textContent = 'Disconnected';
                    statusDiv.className = 'status';
                    disconnectContainer.innerHTML = '';
                });
                
                disconnectContainer.appendChild(disconnectButton);
                
            } catch (e) {
                statusDiv.textContent = 'Error: ' + e.message;
                statusDiv.className = 'status error';
            }
        });
    </script>
</body>
</html>
    """
    return HTMLResponse(html_content)


def create_starlette_app(mcp_server: Server, *, debug: bool = False) -> Starlette:
    """Create a Starlette application that can server the provied mcp server with SSE.

    This sets up the HTTP routes and SSE connection handling.
    """
    # Create an SSE transport with a path for messages
    sse = SseServerTransport("/messages/")

    async def handle_sse(request: Request) -> None:
        async with sse.connect_sse(
            request.scope,
            request.receive,
            request._send,
        ) as (read_stream, write_stream):
            await mcp_server.run(
                read_stream,
                write_stream,
                mcp_server.create_initialization_options(),
            )

    return Starlette(
        debug=debug,
        routes=[
            Route("/", endpoint=homepage),
            Route("/health", endpoint=healthcheck),
            Route("/version", endpoint=version),
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
        ],
    )


def main():
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run MCP SSE-based server")
    parser.add_argument(
        "--transport",
        type=str,
        choices=["stdio", "sse"],
        default="stdio",
        help="Transport mechanism to use ('stdio' for Claude, 'sse' for NANDA)",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on")
    args = parser.parse_args()

    # Create and run the Starlette application
    if args.transport == "sse":
        mcp_server = mcp._mcp_server
        starlette_app = create_starlette_app(mcp_server, debug=True)
        uvicorn.run(starlette_app, host=args.host, port=args.port)
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
