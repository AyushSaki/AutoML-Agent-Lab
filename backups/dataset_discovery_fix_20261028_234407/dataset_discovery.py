import json
import urllib.parse
import urllib.request
import urllib.error
import ssl
from pathlib import Path
import pandas as pd

API_LIST = "https://archive.ics.uci.edu/api/datasets/list"
API_DATASET = "https://archive.ics.uci.edu/api/dataset"
UCI_PAGE = "https://archive.ics.uci.edu/datasets"


def _get_json(url, timeout=20):
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "AutoML-Agent-Lab/1.0 (local research tool)"}
    )
    with urllib.request.urlopen(
        request, timeout=timeout, context=ssl.create_default_context()
    ) as response:
        if response.status != 200:
            raise RuntimeError(f"Repository returned HTTP {response.status}.")
        if response.headers.get("Content-Length"):
            if int(response.headers["Content-Length"]) > 10_000_000:
                raise RuntimeError("Repository metadata response was unexpectedly large.")
        return json.loads(response.read(10_000_001))


def search_uci(query, limit=20):
    query = query.strip()
    if len(query) < 2:
        raise ValueError("Enter at least two characters to search.")
    if len(query) > 120:
        raise ValueError("Search query must be 120 characters or fewer.")

    params = urllib.parse.urlencode({
        "filter": "python",
        "search": query
    })
    payload = _get_json(API_LIST + "?" + params)
    if payload.get("status") != 200:
        raise RuntimeError(payload.get("message", "UCI search failed."))

    results = []
    for item in payload.get("data", [])[:limit]:
        dataset_id = item.get("id")
        if dataset_id is None:
            dataset_id = item.get("uci_id")
        if dataset_id is None:
            continue
        results.append({
            "id": int(dataset_id),
            "name": str(item.get("name") or "Unnamed dataset"),
            "description": str(item.get("description") or item.get("abstract") or ""),
            "repository_url": f"{UCI_PAGE}",
            "task": item.get("task") or item.get("tasks"),
            "instances": item.get("num_instances"),
            "features": item.get("num_features"),
        })
    return results


def get_uci_metadata(dataset_id):
    if not str(dataset_id).isdigit() or int(dataset_id) < 1:
        raise ValueError("Invalid UCI dataset ID.")

    payload = _get_json(API_DATASET + "?" + urllib.parse.urlencode({"id": int(dataset_id)}))
    if payload.get("status") != 200:
        raise RuntimeError(payload.get("message", "Dataset metadata was not found."))

    metadata = payload.get("data", {})
    if not metadata:
        raise RuntimeError("UCI returned no dataset metadata.")

    # Keep only JSON-compatible public metadata.
    clean = {}
    for key, value in metadata.items():
        try:
            json.dumps(value)
            clean[key] = value
        except (TypeError, ValueError):
            clean[key] = str(value)

    clean["metadata_source"] = "UCI Machine Learning Repository API"
    clean["metadata_verified_at_fetch"] = True
    clean["license_status"] = (
        "Metadata explicitly provides a licence; verify its terms before reuse."
        if any("licen" in key.lower() for key in clean)
        else "Licence not explicitly identified in the returned metadata; check the dataset page and terms before reuse."
    )
    return clean


def download_uci_csv(dataset_id, output_dir):
    metadata = get_uci_metadata(dataset_id)
    data_url = metadata.get("data_url")

    if not data_url:
        raise RuntimeError(
            "This dataset has no standardized downloadable data URL. "
            "Open its repository page and follow the available download instructions."
        )
    if not isinstance(data_url, str) or not data_url.startswith("https://"):
        raise RuntimeError("The repository did not provide an acceptable HTTPS data URL.")

    request = urllib.request.Request(
        data_url,
        headers={"User-Agent": "AutoML-Agent-Lab/1.0"}
    )
    with urllib.request.urlopen(
        request, timeout=45, context=ssl.create_default_context()
    ) as response:
        content = response.read(50 * 1024 * 1024 + 1)

    if len(content) > 50 * 1024 * 1024:
        raise RuntimeError("Dataset exceeds the 50 MB download limit for this version.")
    if not content:
        raise RuntimeError("The repository returned an empty file.")

    from io import BytesIO
    try:
        df = pd.read_csv(BytesIO(content), low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(BytesIO(content), encoding="latin-1", low_memory=False)

    if df.empty or len(df.columns) == 0:
        raise RuntimeError("The returned file could not be interpreted as a non-empty CSV.")

    safe_name = "".join(
        char if char.isalnum() or char in "-_." else "_"
        for char in str(metadata.get("name", f"uci_{dataset_id}"))
    ).strip("._")
    if not safe_name:
        safe_name = f"uci_{dataset_id}"

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"uci_{int(dataset_id)}_{safe_name}.csv"
    metadata_path = output_dir / f"uci_{int(dataset_id)}_{safe_name}_metadata.json"

    # Save only after successful parsing.
    output_path.write_bytes(content)
    metadata["downloaded_data_url"] = data_url
    metadata["local_filename"] = output_path.name
    metadata["parsed_rows"] = int(df.shape[0])
    metadata["parsed_columns"] = int(df.shape[1])
    metadata["file_format_check"] = "Parsed as CSV with pandas"
    metadata["data_sha256"] = __import__("hashlib").sha256(content).hexdigest()
    metadata["reuse_warning"] = (
        "Successful download and CSV parsing do not establish permission to reuse. "
        "Confirm the licence, attribution requirements, privacy restrictions, and intended use."
    )
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

    return {
        "metadata": metadata,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": [str(c) for c in df.columns[:100]],
        "local_path": str(output_path),
        "metadata_path": str(metadata_path),
        "preview": df.head(5).fillna("").astype(str).to_dict(orient="records"),
    }
