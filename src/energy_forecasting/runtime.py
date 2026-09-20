"""Shared plumbing for the thin scripts in scripts/: argument parsing, config loading, job task values."""

from __future__ import annotations

import argparse

from loguru import logger

from energy_forecasting.config import ProjectConfig, Tags


def parse_args(extra: list[tuple[str, dict]] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", required=True, help="Bundle root in the workspace (${workspace.root_path})")
    parser.add_argument("--env", required=True, choices=["dev", "uat", "prod"])
    parser.add_argument("--git_sha", default="local")
    parser.add_argument("--branch", default="local")
    parser.add_argument("--job_run_id", default="manual")
    for name, kwargs in extra or []:
        parser.add_argument(name, **kwargs)
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> ProjectConfig:
    cfg = ProjectConfig.from_yaml(f"{args.root_path}/files/project_config.yml", env=args.env)
    logger.info(f"env={cfg.env} catalog={cfg.catalog} schema={cfg.schema_name}")
    return cfg


def tags_from(args: argparse.Namespace) -> Tags:
    return Tags(git_sha=args.git_sha, branch=args.branch, job_run_id=str(args.job_run_id))


def set_task_value(spark, key: str, value) -> None:
    """Pass a value to later tasks in the same job (read by condition tasks)."""
    from pyspark.dbutils import DBUtils

    DBUtils(spark).jobs.taskValues.set(key=key, value=value)
    logger.info(f"task value {key}={value}")
