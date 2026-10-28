from pathlib import Path
from datetime import datetime
import shutil
import sys

root = Path.cwd()
app_file = root / "app.py"
templates = root / "templates"
ml_dir = root / "ml"

if not app_file.exists():
    sys.exit("ERROR: app.py not found. Run from AutoML-Agent-Lab.")

source = app_file.read_text()
if "/dataset-discovery" in source:
    sys.exit("Dataset Discovery route already exists. No changes made.")

startup = "if __name__ == '__main__':"
if startup not in source:
    startup = 'if __name__ == "__main__":'
if startup not in source:
    sys.exit("Cannot locate app startup block safely. No files changed.")

backup = root / "backups" / ("dataset_discovery_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
backup.mkdir(parents=True, exist_ok=True)
shutil.copy2(app_file, backup / "app.py")

base_file = templates / "base.html"
if base_file.exists():
    shutil.copy2(base_file, backup / "base.html")

templates.mkdir(exist_ok=True)
ml_dir.mkdir(exist_ok=True)
(root / "data" / "discovered_datasets").mkdir(parents=True, exist_ok=True)
(root / "reports" / "dataset_discovery").mkdir(parents=True, exist_ok=True)

module = r'''
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
'''

(ml_dir / "dataset_discovery.py").write_text(module.strip() + "\n")

template = r'''{% extends "base.html" %}
{% block title %}Dataset Discovery | AutoML-Agent-Lab{% endblock %}
{% block content %}
<div class="container py-4">
  <h1>Verified Dataset Discovery</h1>
  <p>Search the live UCI Machine Learning Repository, inspect returned metadata, and download supported datasets for local evaluation.</p>

  <div class="alert alert-info">
    Repository metadata and successful downloads can be checked automatically. Licence permission and suitability for your client must still be reviewed.
  </div>

  <form method="get" action="{{ url_for('dataset_discovery') }}" class="card p-4 mb-4">
    <label class="form-label" for="q">Project topic or dataset keywords</label>
    <input class="form-control mb-3" id="q" name="q" value="{{ query }}"
           placeholder="Example: house prices, customer churn, fraud detection" required>
    <button class="btn btn-primary" type="submit">Search UCI Repository</button>
  </form>

  {% if error %}
  <div class="alert alert-danger">{{ error }}</div>
  {% endif %}

  {% if results %}
  <h2>Search results</h2>
  <p class="text-muted">Results are returned by the live repository API. Search ranking is not a suitability recommendation.</p>

  {% for item in results %}
  <div class="card p-3 mb-3">
    <h3>{{ item.name }}</h3>
    <p>{{ item.description or 'No short description was returned by the search endpoint.' }}</p>
    <p><strong>UCI dataset ID:</strong> {{ item.id }}</p>
    {% if item.task %}<p><strong>Repository task metadata:</strong> {{ item.task }}</p>{% endif %}
    {% if item.instances %}<p><strong>Reported instances:</strong> {{ item.instances }}</p>{% endif %}
    {% if item.features %}<p><strong>Reported features:</strong> {{ item.features }}</p>{% endif %}
    <p><a href="https://archive.ics.uci.edu/dataset/{{ item.id }}" target="_blank" rel="noopener">Open dataset page</a></p>
    <form method="post" action="{{ url_for('dataset_discovery_download') }}">
      <input type="hidden" name="dataset_id" value="{{ item.id }}">
      <button class="btn btn-outline-primary" type="submit">Download and validate CSV</button>
    </form>
  </div>
  {% endfor %}
  {% elif searched %}
  <p>No results returned. Try broader keywords or check the repository connection.</p>
  {% endif %}

  {% if downloaded %}
  <div class="card p-4 mt-4">
    <h2>Dataset downloaded and parsed</h2>
    <p><strong>Name:</strong> {{ downloaded.metadata.get('name', 'Unknown') }}</p>
    <p><strong>Rows parsed:</strong> {{ downloaded.rows }}</p>
    <p><strong>Columns parsed:</strong> {{ downloaded.columns }}</p>
    <p><strong>Local file:</strong> <code>{{ downloaded.local_path }}</code></p>
    <p><strong>Metadata JSON:</strong> <code>{{ downloaded.metadata_path }}</code></p>
    <p><strong>Licence review:</strong> {{ downloaded.metadata.license_status }}</p>
    <p><strong>Data URL:</strong> <a href="{{ downloaded.metadata.downloaded_data_url }}" target="_blank" rel="noopener">Repository data URL</a></p>
    <p class="text-muted">A successful download confirms that the file was fetched and parsed, not that it is suitable for your project or licensed for every use.</p>

    <h3>Column preview</h3>
    <p>{{ downloaded.column_names | join(', ') }}</p>

    <h3>First five rows</h3>
    {% if downloaded.preview %}
    <div class="table-responsive">
      <table class="table table-sm table-striped">
        <thead><tr>{% for col in downloaded.column_names[:12] %}<th>{{ col }}</th>{% endfor %}</tr></thead>
        <tbody>
        {% for row in downloaded.preview %}
          <tr>{% for col in downloaded.column_names[:12] %}<td>{{ row.get(col, '') }}</td>{% endfor %}</tr>
        {% endfor %}
        </tbody>
      </table>
    </div>
    {% endif %}
    <p><strong>SHA-256:</strong> <code>{{ downloaded.metadata.data_sha256 }}</code></p>
    <p>{{ downloaded.metadata.reuse_warning }}</p>
  </div>
  {% endif %}
</div>
{% endblock %}
'''

(templates / "dataset_discovery.html").write_text(template.strip() + "\n")

routes = r'''

@app.route("/dataset-discovery", methods=["GET"])
def dataset_discovery():
    from flask import request, render_template
    from ml.dataset_discovery import search_uci

    query = request.args.get("q", "").strip()
    results = []
    error = None
    searched = bool(query)

    if query:
        try:
            results = search_uci(query, limit=20)
        except Exception as exc:
            error = (
                "Could not search the UCI repository. Check your internet connection "
                "and try again. Details: " + str(exc)
            )

    return render_template(
        "dataset_discovery.html",
        query=query,
        results=results,
        error=error,
        searched=searched,
        downloaded=None,
    )


@app.route("/dataset-discovery/download", methods=["POST"])
def dataset_discovery_download():
    from flask import request, render_template
    from ml.dataset_discovery import download_uci_csv, search_uci

    dataset_id = request.form.get("dataset_id", "").strip()
    results = []
    downloaded = None
    error = None

    try:
        if not dataset_id.isdigit():
            raise ValueError("Invalid dataset ID.")
        output_dir = Path(__file__).resolve().parent / "data" / "discovered_datasets"
        downloaded = download_uci_csv(int(dataset_id), output_dir)
    except Exception as exc:
        error = "Download or validation failed: " + str(exc)

    return render_template(
        "dataset_discovery.html",
        query="",
        results=results,
        error=error,
        searched=False,
        downloaded=downloaded,
    )
'''

source = source.replace(startup, routes + "\n\n" + startup, 1)

if "from pathlib import Path" not in source:
    source = "from pathlib import Path\n" + source

app_file.write_text(source)

if base_file.exists():
    html = base_file.read_text()
    if "dataset-discovery" not in html:
        link = '<a class="nav-link" href="{{ url_for(\'dataset_discovery\') }}">Discover Datasets</a>\n'
        if "</nav>" in html:
            html = html.replace("</nav>", link + "</nav>", 1)
        else:
            print("NOTE: Navbar not detected. Open /dataset-discovery directly.")
        base_file.write_text(html)

print("Dataset Discovery Engine installed.")
print("Backup:", backup)
print("Route: /dataset-discovery")
