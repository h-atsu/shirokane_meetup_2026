"""Generate a reproducible synthetic content-delivery instance."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import numpy as np
import polars as pl
import typer

LAYOUT_NAMES = np.array(["compact", "standard", "rich"])
SEGMENTS = 8
EXPECTED_DELIVERIES_PER_USER = 5


def _generate_tables(
    users: int,
    contents: int,
    seed: int,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    rng = np.random.default_rng(seed)

    layout = rng.choice(3, size=users, p=[0.35, 0.45, 0.20])
    base_capacity = np.array([10, 14, 18], dtype=np.int16)
    capacity = np.clip(
        base_capacity[layout] + rng.integers(-2, 3, size=users), 8, 20
    ).astype(np.int16)

    weight = rng.choice(
        np.arange(1, 6, dtype=np.int16),
        size=contents,
        p=[0.25, 0.28, 0.23, 0.15, 0.09],
    )
    user_segment = rng.integers(0, SEGMENTS, size=users, dtype=np.uint16)
    appeal = rng.normal(0.0, 2.0, size=contents).astype(np.float32)
    affinity = rng.normal(0.0, 3.0, size=(SEGMENTS, contents))
    score = (
        8.0
        + appeal[None, :]
        + affinity[user_segment]
        + rng.normal(0.0, 3.0, size=(users, contents))
    )
    score = np.maximum(score, 0.25).astype(np.float32)

    expected_total_deliveries = users * EXPECTED_DELIVERIES_PER_USER
    minimum = np.full(contents, expected_total_deliveries // contents, dtype=np.int64)
    minimum[: expected_total_deliveries % contents] += 1

    user_ids = np.arange(users, dtype=np.uint32)
    content_id_dtype = np.uint16 if contents <= np.iinfo(np.uint16).max else np.uint32
    content_ids = np.arange(contents, dtype=content_id_dtype)

    users_table = pl.DataFrame(
        {
            "user_id": user_ids,
            "layout": LAYOUT_NAMES[layout],
            "segment_id": user_segment,
            "capacity": capacity,
        }
    )
    user_contents_table = pl.DataFrame(
        {
            "user_id": np.repeat(user_ids, contents),
            "content_id": np.tile(content_ids, users),
            "score": score.ravel(),
        }
    )
    contents_table = pl.DataFrame(
        {
            "content_id": content_ids,
            "weight": weight,
            "appeal": appeal,
            "minimum_delivery_count": minimum,
        }
    )
    return users_table, user_contents_table, contents_table


def main(
    instance_name: Annotated[
        str | None,
        typer.Argument(
            help="Instance directory name. Generated from the options when omitted."
        ),
    ] = None,
    users: Annotated[
        int, typer.Option(min=1, help="Number of users to generate.")
    ] = 50_000,
    contents: Annotated[
        int, typer.Option(min=1, help="Number of contents to generate.")
    ] = 100,
    seed: Annotated[int, typer.Option(help="Random seed.")] = 0,
    output_root: Annotated[
        Path,
        typer.Option(
            file_okay=False,
            dir_okay=True,
            help="Directory under which the instance directory is created.",
        ),
    ] = Path("data"),
) -> None:
    """Generate Parquet source data for a content-delivery experiment."""
    instance_name = instance_name or f"users_{users}_contents_{contents}_seed_{seed}"
    destination = output_root / instance_name

    typer.echo(f"Generating {users:,} users x {contents:,} contents (seed={seed})...")
    users_table, user_contents_table, contents_table = _generate_tables(
        users=users,
        contents=contents,
        seed=seed,
    )

    destination.mkdir(parents=True, exist_ok=True)
    users_table.write_parquet(destination / "users.parquet", compression="zstd")
    user_contents_table.write_parquet(
        destination / "user_contents.parquet", compression="zstd"
    )
    contents_table.write_parquet(destination / "contents.parquet", compression="zstd")
    metadata = {
        "instance_name": instance_name,
        "users": users,
        "contents": contents,
        "segments": SEGMENTS,
        "expected_deliveries_per_user": EXPECTED_DELIVERIES_PER_USER,
        "expected_total_deliveries": users * EXPECTED_DELIVERIES_PER_USER,
        "seed": seed,
    }
    (destination / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    typer.echo(f"Wrote instance to {destination}")
    for path in sorted(destination.iterdir()):
        typer.echo(f"  {path.name}: {path.stat().st_size / 1_048_576:.2f} MiB")


if __name__ == "__main__":
    typer.run(main)
