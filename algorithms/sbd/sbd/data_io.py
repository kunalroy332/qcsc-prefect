# Workflow for observability demo on Miyabi
#
# Author: Naoki Kanazawa (knzwnao@jp.ibm.com)

import io
import json
import os
import uuid
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import numpy as np
from prefect.artifacts import TableArtifact
from prefect.client.orchestration import get_client
from prefect.client.schemas.actions import ArtifactUpdate
from prefect.client.schemas.filters import (
    ArtifactFilter,
    ArtifactFilterKey,
    FlowRunFilter,
    FlowRunFilterId,
)
from prefect.context import FlowRunContext
from prefect.settings import get_current_settings

try:
    from prefect_aws.s3 import S3Bucket
except Exception:
    S3Bucket = None  # type: ignore


def _flow_scoped_subdir() -> str:
    ctx = FlowRunContext.get()
    if ctx and ctx.flow and ctx.flow_run:
        return os.path.join(ctx.flow.name, ctx.flow_run.name)
    return os.path.join("no-context", uuid.uuid4().hex)


def _local_storage_dir() -> Path:
    """Local fallback for persisted arrays (sample pools, SQD reports).

    ``SBD_LOCAL_STORAGE_DIR`` overrides it, e.g. one directory per run when PREFECT_HOME must stay
    the user's own (Prefect Cloud profile). Default: ``$PREFECT_HOME/storage``.
    """
    override = os.environ.get("SBD_LOCAL_STORAGE_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    settings = get_current_settings()
    return Path(settings.home) / "storage"


def save_ndarray(
    file_prefix: str,
    **arrays: np.ndarray,
) -> str:

    subdir = _flow_scoped_subdir()

    # --- Try S3 first ---
    if S3Bucket is not None:
        try:
            s3_client = S3Bucket.load("s3-sqd")
            s3_client.bucket_folder = subdir

            with io.BytesIO() as buf:
                np.savez_compressed(buf, **arrays, allow_pickle=False)
                buf.seek(0)
                key = s3_client.upload_from_file_object(
                    buf,
                    to_path=f"{file_prefix}_{uuid.uuid4().hex}.npz",
                )
            return str(key)
        except Exception:
            pass

    # --- Fallback to local ---
    base = _local_storage_dir() / "sqd_data" / subdir
    base.mkdir(parents=True, exist_ok=True)
    local_path = base / f"{file_prefix}_{uuid.uuid4().hex}.npz"

    with open(local_path, "wb") as f:
        with io.BytesIO() as buf:
            np.savez_compressed(buf, **arrays, allow_pickle=False)
            buf.seek(0)
            f.write(buf.read())

    return f"file://{local_path}"


def _local_path(file_path: str) -> str | None:
    """The local file a saved-artifact path names, or None for an S3 key.

    Local: ``file://<path>``, an absolute or ``~`` path, or any existing file. S3 keys are
    bucket-relative, so a plain absolute path (e.g. typed into saved_samples) never goes to S3.
    """
    if file_path.startswith("file://"):
        return file_path[len("file://") :]
    path = os.path.expanduser(file_path)
    if os.path.isabs(path) or os.path.isfile(path):
        return path
    return None


def load_ndarray(
    file_path: str,
    key: str,
) -> np.ndarray:
    # --- Local ---
    local_path = _local_path(file_path)
    if local_path is not None:
        with open(local_path, "rb") as f:
            data = f.read()
        with io.BytesIO(data) as buf:
            buf.seek(0)
            arr = np.load(buf, allow_pickle=False).get(key)
        assert arr is not None
        return arr

    # --- S3 (backward compatible) ---
    try:
        s3_client = S3Bucket.load("s3-sqd")
    except Exception as exc:
        raise FileNotFoundError(
            f"{file_path}: not a local file and no 's3-sqd' block to read it from S3 "
            "(local pools: an absolute path or file://<path>)"
        ) from exc
    with io.BytesIO() as buf:
        s3_client.download_object_to_file_object(file_path, buf)
        buf.seek(0)
        arr = np.load(buf, allow_pickle=False).get(key)
    assert arr is not None
    return arr


def extend_table_artifact(
    artifact_key: str,
    new_table: list[dict[str, Any]] | list[list[Any]],
    index: int = 0,
) -> UUID:
    ctx_flow = FlowRunContext.get()

    with get_client(sync_client=True) as client:
        artifacts = client.read_artifacts(
            artifact_filter=ArtifactFilter(
                key=ArtifactFilterKey(any_=[artifact_key]),
            ),
            flow_run_filter=FlowRunFilter(
                id=FlowRunFilterId(any_=[ctx_flow.flow_run.id]),
            ),
        )
        artifact = artifacts[index]
        current_data = json.loads(artifact.data)

        assert isinstance(current_data, list)
        current_data.extend(new_table)

        update = ArtifactUpdate(data=cast(str, TableArtifact(table=current_data).format()))
        client.update_artifact(
            artifact_id=artifact.id,
            artifact=update,
        )
    return artifact.id
