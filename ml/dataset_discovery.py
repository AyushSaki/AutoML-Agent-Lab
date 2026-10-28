"""
Dataset Discovery Engine.

Searches the UCI Machine Learning Repository and returns
multiple ranked dataset candidates for an AutoML project.

This module is intentionally independent from the training
pipeline so existing model training remains untouched.
"""

import json
import hashlib
import ssl
import urllib.error
import urllib.parse
import urllib.request
from io import BytesIO
from pathlib import Path

import pandas as pd


API_LIST = "https://archive.ics.uci.edu/api/datasets/list"
API_DATASET = "https://archive.ics.uci.edu/api/dataset"

UCI_DATASET_PAGE = (
    "https://archive.ics.uci.edu/dataset/"
)


def _get_json(url, timeout=25):
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "AutoML-Agent-Lab/1.0"
        },
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=timeout,
            context=ssl.create_default_context(),
        ) as response:

            if response.status != 200:
                raise RuntimeError(
                    f"Repository returned HTTP {response.status}."
                )

            content = response.read(15_000_001)

    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not connect to the UCI repository: {exc}"
        ) from exc

    if len(content) > 15_000_000:
        raise RuntimeError(
            "Repository metadata response was unexpectedly large."
        )

    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "The UCI repository returned invalid JSON."
        ) from exc


def _normalise_text(value):
    if value is None:
        return ""

    if isinstance(value, (list, tuple)):
        return " ".join(
            str(item) for item in value
        )

    if isinstance(value, dict):
        return " ".join(
            f"{key} {value}"
            for key, value in value.items()
        )

    return str(value)


def _dataset_id(item):
    value = (
        item.get("id")
        or item.get("uci_id")
        or item.get("dataset_id")
    )

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _dataset_description(item):
    return (
        item.get("description")
        or item.get("abstract")
        or item.get("summary")
        or ""
    )


def _dataset_task(item):
    return (
        item.get("task")
        or item.get("tasks")
        or item.get("default_task")
        or ""
    )


def _dataset_search_text(item):
    return " ".join(
        [
            _normalise_text(item.get("name")),
            _normalise_text(_dataset_description(item)),
            _normalise_text(item.get("area")),
            _normalise_text(item.get("task")),
            _normalise_text(item.get("tasks")),
        ]
    ).lower()


def _score_dataset(item, query):
    """
    Relevance score.

    Exact phrase > name matches > token matches > description matches.
    """

    query = query.lower().strip()

    if not query:
        return 0

    name = _normalise_text(item.get("name")).lower()
    description = _normalise_text(
        _dataset_description(item)
    ).lower()
    area = _normalise_text(
        item.get("area")
    ).lower()

    text = " ".join(
        [name, description, area]
    )

    tokens = [
        token
        for token in query.replace(",", " ").split()
        if len(token) >= 2
    ]

    score = 0

    # Exact phrase.
    if query in name:
        score += 100

    if query in description:
        score += 40

    # Individual tokens.
    for token in tokens:

        if token in name:
            score += 25

        elif token in description:
            score += 10

        elif token in area:
            score += 8

        elif token in text:
            score += 3

    return score


def _search_endpoint(query=None):
    params = {
        "filter": "python",
    }

    if query:
        params["search"] = query

    url = (
        API_LIST
        + "?"
        + urllib.parse.urlencode(params)
    )

    payload = _get_json(url)

    if payload.get("status") != 200:
        raise RuntimeError(
            payload.get(
                "message",
                "UCI dataset search failed.",
            )
        )

    data = payload.get("data", [])

    if not isinstance(data, list):
        return []

    return data


def search_uci(query, limit=20):
    """
    Search UCI and return multiple ranked candidates.

    Strategy:
    1. Search the complete phrase.
    2. Search individual meaningful words.
    3. Merge results by dataset ID.
    4. Rank locally.
    5. Return up to `limit` datasets.
    """

    query = str(query or "").strip()

    if len(query) < 2:
        raise ValueError(
            "Enter at least two characters to search."
        )

    if len(query) > 150:
        raise ValueError(
            "Search query must be 150 characters or fewer."
        )

    # -----------------------------------------------------
    # SEARCH TERMS
    # -----------------------------------------------------

    terms = [query]

    tokens = [
        token.strip(".,:;!?()[]{}")
        for token in query.split()
    ]

    meaningful_tokens = [
        token
        for token in tokens
        if len(token) >= 3
    ]

    # Search individual words as well.
    for token in meaningful_tokens[:6]:
        if token.lower() not in {
            term.lower()
            for term in terms
        }:
            terms.append(token)

    # -----------------------------------------------------
    # MERGE RESULTS
    # -----------------------------------------------------

    datasets = {}

    for term in terms:

        try:
            rows = _search_endpoint(term)
        except Exception:
            # A failed secondary search should not destroy
            # the successful primary search.
            continue

        for item in rows:

            dataset_id = _dataset_id(item)

            if dataset_id is None:
                continue

            existing = datasets.get(dataset_id)

            if existing is None:
                datasets[dataset_id] = dict(item)

            else:
                # Keep the richer description/metadata.
                for key, value in item.items():

                    if not existing.get(key) and value:
                        existing[key] = value

    # -----------------------------------------------------
    # FALLBACK
    # -----------------------------------------------------

    # If the phrase is too restrictive, retrieve the
    # available Python-importable catalogue and rank it
    # locally. This guarantees that a topic such as
    # "house price prediction" can surface multiple
    # candidates even when the exact phrase matches only
    # one repository title.
    if len(datasets) < min(8, limit):

        try:
            catalogue = _search_endpoint(None)

            for item in catalogue:

                dataset_id = _dataset_id(item)

                if dataset_id is None:
                    continue

                if dataset_id not in datasets:
                    datasets[dataset_id] = dict(item)

        except Exception:
            pass

    # -----------------------------------------------------
    # RANK
    # -----------------------------------------------------

    ranked = []

    for item in datasets.values():

        score = _score_dataset(
            item,
            query,
        )

        ranked.append(
            {
                "id": _dataset_id(item),
                "name": str(
                    item.get(
                        "name",
                        "Unnamed dataset",
                    )
                ),
                "description": str(
                    _dataset_description(item)
                ),
                "task": _dataset_task(item),
                "area": item.get("area"),
                "instances": (
                    item.get("num_instances")
                    or item.get("instances")
                ),
                "features": (
                    item.get("num_features")
                    or item.get("features")
                ),
                "feature_types": item.get(
                    "feature_types"
                ),
                "has_missing_values": item.get(
                    "has_missing_values"
                ),
                "target_col": item.get(
                    "target_col"
                ),
                "relevance_score": score,
                "repository_url": (
                    UCI_DATASET_PAGE
                    + str(_dataset_id(item))
                ),
            }
        )

    # Remove completely irrelevant results when
    # meaningful matches exist.
    matched = [
        item
        for item in ranked
        if item["relevance_score"] > 0
    ]

    if matched:
        ranked = matched

    ranked.sort(
        key=lambda item: (
            item["relevance_score"],
            item["instances"] or 0,
        ),
        reverse=True,
    )

    return ranked[:limit]


def get_uci_metadata(dataset_id):
    if not str(dataset_id).isdigit():
        raise ValueError(
            "Invalid UCI dataset ID."
        )

    dataset_id = int(dataset_id)

    if dataset_id < 1:
        raise ValueError(
            "Invalid UCI dataset ID."
        )

    payload = _get_json(
        API_DATASET
        + "?"
        + urllib.parse.urlencode(
            {"id": dataset_id}
        )
    )

    if payload.get("status") != 200:
        raise RuntimeError(
            payload.get(
                "message",
                "Dataset metadata was not found.",
            )
        )

    metadata = payload.get("data", {})

    if not metadata:
        raise RuntimeError(
            "UCI returned no dataset metadata."
        )

    clean = {}

    for key, value in metadata.items():

        try:
            json.dumps(value)
            clean[key] = value

        except (TypeError, ValueError):
            clean[key] = str(value)

    clean["metadata_source"] = (
        "UCI Machine Learning Repository API"
    )

    clean["repository_url"] = (
        UCI_DATASET_PAGE
        + str(dataset_id)
    )

    clean["license_status"] = (
        "Check the official dataset page and licence "
        "before reuse. Successful download does not "
        "automatically grant permission for every use."
    )

    return clean


def download_uci_csv(
    dataset_id,
    output_dir,
):
    metadata = get_uci_metadata(
        dataset_id
    )

    data_url = metadata.get(
        "data_url"
    )

    if not data_url:
        raise RuntimeError(
            "This dataset does not expose a standardized "
            "CSV download through the UCI API."
        )

    if (
        not isinstance(data_url, str)
        or not data_url.startswith("https://")
    ):
        raise RuntimeError(
            "UCI did not provide an acceptable HTTPS "
            "download URL."
        )

    request = urllib.request.Request(
        data_url,
        headers={
            "User-Agent": "AutoML-Agent-Lab/1.0"
        },
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=60,
            context=ssl.create_default_context(),
        ) as response:

            content = response.read(
                50 * 1024 * 1024 + 1
            )

    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Dataset download failed: {exc}"
        ) from exc

    if len(content) > 50 * 1024 * 1024:
        raise RuntimeError(
            "Dataset exceeds the 50 MB download limit."
        )

    if not content:
        raise RuntimeError(
            "The downloaded dataset is empty."
        )

    try:
        df = pd.read_csv(
            BytesIO(content),
            low_memory=False,
        )

    except UnicodeDecodeError:
        df = pd.read_csv(
            BytesIO(content),
            encoding="latin-1",
            low_memory=False,
        )

    except Exception as exc:
        raise RuntimeError(
            f"Downloaded file could not be parsed as CSV: {exc}"
        ) from exc

    if df.empty:
        raise RuntimeError(
            "The downloaded dataset contains no rows."
        )

    if len(df.columns) == 0:
        raise RuntimeError(
            "The downloaded dataset contains no columns."
        )

    output_dir = Path(
        output_dir
    ).resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_name = str(
        metadata.get(
            "name",
            f"uci_{dataset_id}",
        )
    )

    safe_name = "".join(
        character
        if character.isalnum()
        or character in "-_."
        else "_"
        for character in raw_name
    ).strip("._")

    if not safe_name:
        safe_name = f"uci_{dataset_id}"

    output_path = (
        output_dir
        / f"uci_{dataset_id}_{safe_name}.csv"
    )

    metadata_path = (
        output_dir
        / f"uci_{dataset_id}_{safe_name}_metadata.json"
    )

    output_path.write_bytes(
        content
    )

    metadata[
        "downloaded_data_url"
    ] = data_url

    metadata[
        "local_filename"
    ] = output_path.name

    metadata[
        "parsed_rows"
    ] = int(df.shape[0])

    metadata[
        "parsed_columns"
    ] = int(df.shape[1])

    metadata[
        "column_names"
    ] = [
        str(column)
        for column in df.columns
    ]

    metadata[
        "data_sha256"
    ] = hashlib.sha256(
        content
    ).hexdigest()

    metadata[
        "reuse_warning"
    ] = (
        "Successful download and CSV parsing do not "
        "establish permission to reuse. Confirm the "
        "licence, attribution, privacy restrictions, "
        "and intended use."
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return {
        "metadata": metadata,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": [
            str(column)
            for column in df.columns
        ],
        "local_path": str(
            output_path
        ),
        "metadata_path": str(
            metadata_path
        ),
        "preview": (
            df.head(5)
            .fillna("")
            .astype(str)
            .to_dict(
                orient="records"
            )
        ),
    }
