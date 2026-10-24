"""Contrato y escritura comunes a los dos preparadores de datos (versión 2)."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

VERSION = "2.1.0"
ARTICLE_COLUMNS = [
    "article_id", "product_code", "product_type_name", "product_group_name",
    "graphical_appearance_name", "colour_group_name", "perceived_colour_value_name",
    "index_group_name", "section_name", "garment_group_name",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_columns(frame, columns):
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Faltan columnas: {missing}")


def load_articles(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"article_id": "string", "product_code": "string"})
    require_columns(frame, ARTICLE_COLUMNS)
    for name in ["article_id", "product_code"]:
        if not frame[name].str.fullmatch(r"[0-9]+", na=False).all():
            raise ValueError(f"{name}: identificadores vacíos o no numéricos")
    if frame["article_id"].duplicated().any():
        raise ValueError("article_id duplicado en articles.csv")
    # Las categorías ausentes se declaran explícitamente, sin perder artículos.
    for name in ARTICLE_COLUMNS[2:]:
        frame[name] = frame[name].astype("string").fillna("Unknown").replace("", "Unknown")
    if "detail_desc" in frame:
        frame["detail_desc"] = frame["detail_desc"].fillna("")
    return frame


def articles_path(root: Path, raw_dir: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.resolve()
    # En el espacio de trabajo con ambos repos, demanda contiene la copia canónica.
    canonical = root.parent / "fashion-demand-ml/data/raw/articles.csv"
    return canonical if canonical.is_file() else raw_dir / "articles.csv"


def metadata(parameters: dict, inputs: dict[str, Path], sources: list[Path]) -> dict:
    packages = {}
    for name in ["pandas", "numpy", "scikit-learn", "Pillow", "pyarrow"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {
        "pipeline_version": VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "packages": packages,
        "parameters": parameters,
        "inputs": {name: {"path": str(path.resolve()), "sha256": sha256(path)}
                   for name, path in inputs.items()},
        "source_sha256": {p.name: sha256(p) for p in sources},
    }


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def publish(output_dir: Path, tables: dict, reports: dict, parquet=False, preview=False):
    """Publica una ejecución completa y luego cambia CURRENT.json atómicamente.

    Los consumidores leen CURRENT.json. Un fallo deja la ejecución anterior intacta.
    Los CSV antiguos de la raíz no se sobrescriben y no deben seguir usándose.
    """
    if parquet:
        import pyarrow  # noqa: F401: comprobar antes de escribir cualquier salida
    runs = output_dir / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    with tempfile.TemporaryDirectory(prefix=".pending-", dir=runs) as temporary:
        stage = Path(temporary)
        inventory = {}
        for name, frame in tables.items():
            csv_name = name + ".csv.gz"
            frame.to_csv(stage / csv_name, index=False, compression="gzip")
            inventory[name] = {"csv": csv_name, "rows": len(frame),
                               "dtypes": {c: str(t) for c, t in frame.dtypes.items()}}
            if parquet:
                pq_name = name + ".parquet"
                frame.to_parquet(stage / pq_name, index=False)
                inventory[name]["parquet"] = pq_name
        for name, report in reports.items():
            write_json(stage / (name + ".json"), report)
        write_json(stage / "schema.json", inventory)
        # Cada archivo queda identificado para detectar cambios posteriores.
        checksums = {p.name: sha256(p) for p in stage.iterdir() if p.is_file()}
        write_json(stage / "checksums.json", checksums)
        final = runs / run_id
        stage.rename(final)
    pointer = {"run_dir": f"runs/{run_id}", "pipeline_version": VERSION, "preview": preview}
    # Las pruebas parciales nunca sustituyen el resultado completo vigente.
    pointer_name = "PREVIEW.json" if preview else "CURRENT.json"
    temp_pointer = output_dir / ("." + pointer_name + "." + uuid.uuid4().hex)
    write_json(temp_pointer, pointer)
    os.replace(temp_pointer, output_dir / pointer_name)
    return final
