"""Optional real-time endpoint serving the current @champion.

The daily forecast does NOT need this - it scores in batch. An endpoint is for
other applications asking for forecasts on demand. Because the model was logged
with Feature Engineering, the endpoint looks features up from an online store,
so features must be published online first (handled in Step 10).
"""

from __future__ import annotations

from loguru import logger

from energy_forecasting.config import ProjectConfig


def deploy_champion(cfg: ProjectConfig, version: str) -> None:
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.serving import AutoCaptureConfigInput, EndpointCoreConfigInput, ServedEntityInput

    w = WorkspaceClient()
    entities = [
        ServedEntityInput(
            entity_name=cfg.model_name,
            entity_version=version,
            workload_size="Small",
            scale_to_zero_enabled=True,
        )
    ]
    exists = any(e.name == cfg.endpoint_name for e in w.serving_endpoints.list())
    if exists:
        w.serving_endpoints.update_config(name=cfg.endpoint_name, served_entities=entities)
    else:
        w.serving_endpoints.create(
            name=cfg.endpoint_name,
            config=EndpointCoreConfigInput(
                served_entities=entities,
                auto_capture_config=AutoCaptureConfigInput(
                    catalog_name=cfg.catalog, schema_name=cfg.schema_name, table_name_prefix="serving"
                ),
            ),
        )
    logger.info(f"Endpoint {cfg.endpoint_name} -> {cfg.model_name} v{version}")
