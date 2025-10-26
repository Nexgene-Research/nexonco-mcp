from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

from .query import (
    BROWSE_PHENOTYPES_QUERY,
    EVIDENCE_BROWSE_QUERY,
    EVIDENCE_SUMMARY_QUERY,
)


class CivicAPIClient:
    """
    Client for interacting with the CIViC (Clinical Interpretation of Variants in Cancer) GraphQL API.
    Provides methods to browse phenotypes, retrieve evidence, and source details in bulk.
    """

    def __init__(self, cookies=None):
        """
        Initialize the CIViC API client.

        Args:
            cookies (dict, optional): Cookies for authenticated requests.
        """
        self.base_url = "https://civicdb.org/api/graphql"
        self.cookies = cookies or {}
        self.headers = {
            "Accept": "application/json, text/plain, */*",
            "Cache-Control": "no-cache",
        }

    def browse_phenotype(self, phenotype_name=None):
        """
        Retrieve phenotype information from the CIViC API.

        Args:
            phenotype_name (str, optional): Name of the phenotype to browse.

        Returns:
            dict: JSON response containing phenotype data.
        """
        variables = {"phenotypeName": phenotype_name}
        payload = {
            "operationName": "BrowsePhenotypes",
            "variables": variables,
            "query": BROWSE_PHENOTYPES_QUERY,
        }
        result = self._send_request(payload)
        return result["data"]["browsePhenotypes"]["edges"][0]["node"]

    def get_sources(self, evidence_id_list):
        """
        Fetch source information for multiple evidence items in parallel.

        Args:
            evidence_id_list (list of int): List of evidence IDs.

        Returns:
            list of dict: List of source information for each evidence item.
        """
        payloads = [
            {
                "operationName": "EvidenceSummary",
                "variables": {"evidenceId": eid},
                "query": EVIDENCE_SUMMARY_QUERY,
            }
            for eid in evidence_id_list
        ]
        results = self._send_parallel_requests(payloads)
        return [res["data"]["evidenceItem"]["source"] for res in results]

    def search_evidence(
        self,
        disease_name=None,
        therapy_name=None,
        molecular_profile_name=None,
        phenotype_name=None,
        filter_strong_evidence=False,
        evidence_type=None,
        evidence_direction=None,
    ):
        """
        Search for evidence items based on filters like disease, therapy, and molecular profile.

        Args:
            disease_name (str, optional): Disease name to filter.
            therapy_name (str, optional): Therapy name to filter.
            molecular_profile_name (str, optional): Molecular profile name to filter.
            phenotype_name (str, optional): Phenotype name to filter.
            filter_strong_evidence (bool): Whether to include only strong evidence (rating > 3).
            evidence_type (str, optional): Type of evidence ("PREDICTIVE" or "DIAGNOSTIC" or "PROGNOSTIC" or "PREDISPOSING" or "FUNCTIONAL").
            evidence_direction (str, optional): Direction of evidence (SUPPORTS or DOES_NOT_SUPPORT).

        Returns:
            pd.DataFrame: DataFrame containing filtered evidence items and source information.
        """
        variables = {"sortBy": {"column": "EVIDENCE_RATING", "direction": "DESC"}}

        if evidence_type in [
            "PREDICTIVE",
            "DIAGNOSTIC",
            "PROGNOSTIC",
            "PREDISPOSING",
            "FUNCTIONAL",
        ]:
            variables["evidenceType"] = evidence_type
            if evidence_direction in ["SUPPORTS", "DOES_NOT_SUPPORT"]:
                variables["evidenceDirection"] = evidence_direction

        variables["status"] = "ACCEPTED" if filter_strong_evidence else "NON_REJECTED"

        if disease_name:
            variables["diseaseName"] = disease_name
        if therapy_name:
            variables["therapyName"] = therapy_name
        if molecular_profile_name:
            variables["molecularProfileName"] = molecular_profile_name

        phenotype_data = {"id": None, "name": None}
        if phenotype_name:
            phenotype_data = self.browse_phenotype(phenotype_name)
            variables["phenotypeId"] = phenotype_data["id"]

        payload = {
            "operationName": "EvidenceBrowse",
            "variables": variables,
            "query": EVIDENCE_BROWSE_QUERY,
        }

        results = self._send_request(payload)
        results = results["data"]["evidenceItems"]["edges"]

        data = []
        for entry in results:
            result = entry["node"]

            if filter_strong_evidence and entry["evidenceRating"] <= 3:
                continue

            # Safely parse all nested fields with None checks

            # Disease fields
            disease = result.get("disease")
            disease_id = disease.get("id") if disease else None
            disease_name = disease.get("name") if disease else None

            # Therapy fields
            therapies = result.get("therapies", [])
            therapy_ids = "+".join([str(t.get("id", "")) for t in therapies if t]) if therapies else None
            therapy_names = "+".join([t.get("name", "") for t in therapies if t]) if therapies else None

            # Molecular profile fields
            mol_profile = result.get("molecularProfile")
            mol_profile_id = mol_profile.get("id") if mol_profile else None
            mol_profile_name = mol_profile.get("name") if mol_profile else None

            # Gene and variant from parsedName
            gene_id = None
            gene_name = None
            variant_id = None
            variant_name = None

            if mol_profile:
                parsed_name = mol_profile.get("parsedName", [])
                if parsed_name and len(parsed_name) > 0:
                    # First element is usually the gene
                    if isinstance(parsed_name[0], dict) and "id" in parsed_name[0]:
                        gene_id = parsed_name[0].get("id")
                        gene_name = parsed_name[0].get("name")

                    # Second element is usually the variant
                    if len(parsed_name) > 1 and isinstance(parsed_name[1], dict) and "id" in parsed_name[1]:
                        variant_id = parsed_name[1].get("id")
                        variant_name = parsed_name[1].get("name")

            evidence = {
                "id": result.get("id"),
                "name": result.get("name"),
                "disease_id": disease_id,
                "disease_name": disease_name,
                "therapy_ids": therapy_ids,
                "therapy_names": therapy_names,
                "molecular_profile_id": mol_profile_id,
                "molecular_profile_name": mol_profile_name,
                "gene_id": gene_id,
                "gene_name": gene_name,
                "variant_id": variant_id,
                "variant_name": variant_name,
                "phenotype_id": phenotype_data.get("id"),
                "phenotype_name": phenotype_data.get("name"),
                "description": result.get("description"),
                "evidence_type": result.get("evidenceType"),
                "evidence_direction": result.get("evidenceDirection"),
                "evidence_rating": result.get("evidenceRating"),
            }

            data.append(evidence)

        df = pd.DataFrame(data)
        df = df.dropna(subset=["evidence_rating"])

        return df

    def _send_request(self, payload):
        """
        Internal method to send a single request to the CIViC API.

        Args:
            payload (dict): GraphQL query payload.

        Returns:
            dict: Parsed JSON response from the API.
        """
        response = requests.post(
            self.base_url, headers=self.headers, cookies=self.cookies, json=payload
        )

        # Raise exception for HTTP errors
        response.raise_for_status()

        return response.json()

    def _send_parallel_requests(self, payloads, max_workers=12):
        """
        Internal method to send multiple GraphQL requests concurrently.

        Args:
            payloads (list): List of GraphQL payloads.
            max_workers (int): Number of concurrent threads (default 12).

        Returns:
            list of dict: List of API responses for each payload.
        """
        results = []

        def send(payload):
            return self._send_request(payload)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_payload = {executor.submit(send, p): p for p in payloads}
            for future in as_completed(future_to_payload):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append(
                        {"error": str(e), "payload": future_to_payload[future]}
                    )

        return results


    def search_evidence_batch(
        self,
        disease_names=None,
        therapy_names=None,
        molecular_profile_names=None,
        **kwargs,
    ):
        """
        Perform batch searches for multiple diseases, therapies, or molecular profiles.

        This method executes parallel searches when multiple values are provided for any parameter,
        aggregates results, and returns a combined DataFrame.

        Args:
            disease_names (list of str, optional): List of disease names to search.
            therapy_names (list of str, optional): List of therapy names to search.
            molecular_profile_names (list of str, optional): List of molecular profile names to search.
            **kwargs: Additional parameters passed to search_evidence (e.g., evidence_type, filter_strong_evidence).

        Returns:
            dict: Dictionary mapping search parameters to their result DataFrames.
                  Format: {param_value: DataFrame, ...}
        """
        results_by_param = {}

        # Search by diseases
        if disease_names and isinstance(disease_names, list):
            for disease in disease_names:
                df = self.search_evidence(disease_name=disease, **kwargs)
                results_by_param[f"Disease: {disease}"] = df

        # Search by therapies
        if therapy_names and isinstance(therapy_names, list):
            for therapy in therapy_names:
                df = self.search_evidence(therapy_name=therapy, **kwargs)
                results_by_param[f"Therapy: {therapy}"] = df

        # Search by molecular profiles
        if molecular_profile_names and isinstance(molecular_profile_names, list):
            for profile in molecular_profile_names:
                df = self.search_evidence(molecular_profile_name=profile, **kwargs)
                results_by_param[f"Gene/Variant: {profile}"] = df

        return results_by_param

    def compare_therapies_data(
        self, therapy_names, disease_name=None, molecular_profile_name=None, **kwargs
    ):
        """
        Retrieve evidence data for multiple therapies to enable comparison.

        Args:
            therapy_names (list of str): List of therapy names to compare.
            disease_name (str, optional): Disease context for comparison.
            molecular_profile_name (str, optional): Molecular profile context for comparison.
            **kwargs: Additional parameters passed to search_evidence.

        Returns:
            dict: Dictionary mapping therapy names to their evidence DataFrames.
                  Format: {therapy_name: DataFrame, ...}
        """
        therapy_data = {}

        for therapy in therapy_names:
            df = self.search_evidence(
                therapy_name=therapy,
                disease_name=disease_name,
                molecular_profile_name=molecular_profile_name,
                **kwargs,
            )
            therapy_data[therapy] = df

        return therapy_data

    def analyze_molecular_profile_data(
        self, molecular_profile_name, disease_name=None, therapy_name=None, **kwargs
    ):
        """
        Retrieve comprehensive evidence data for a specific gene or variant.

        Args:
            molecular_profile_name (str): Gene or variant name to analyze.
            disease_name (str, optional): Filter by disease context.
            therapy_name (str, optional): Filter by therapy context.
            **kwargs: Additional parameters passed to search_evidence.

        Returns:
            pd.DataFrame: Comprehensive evidence data for the molecular profile.
        """
        return self.search_evidence(
            molecular_profile_name=molecular_profile_name,
            disease_name=disease_name,
            therapy_name=therapy_name,
            **kwargs,
        )


def example_usage():
    """Example of how to use the CivicAPIClient."""
    import json

    client = CivicAPIClient()
    results = client.search_evidence(
        disease_name="cancer", therapy_name="ce", molecular_profile_name="egfr"
    )
    print(results)

    # print(json.dumps(client.browse_phenotype("pain"), indent=2))
    # print(json.dumps(client.get_sources([1572, 1058, 7096]), indent=2))


if __name__ == "__main__":
    example_usage()
