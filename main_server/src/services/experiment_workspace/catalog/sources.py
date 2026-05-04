"""Experiment catalog가 읽는 repo-local source 수집 adapter."""

from __future__ import annotations

import importlib.util
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf


@dataclass(frozen=True, slots=True)
class ExperimentCatalogSource:
    """Hydra config, Python module, repo path 해석을 한곳에 모은다."""

    repo_root: Path

    def load_config_group_item(
        self,
        *,
        relative_dir: str,
        item_name: str,
    ) -> dict[str, object]:
        """Hydra config group item 하나를 직접 읽는다."""

        path = self.repo_root / relative_dir / f"{item_name}.yaml"
        if not path.exists():
            raise ValueError(
                f"Unknown config group item: dir={relative_dir}, item={item_name}."
            )
        return self.load_yaml_mapping(path)

    def load_relative_yaml_mapping(self, relative_path: str) -> dict[str, object]:
        """repo root 기준 상대 경로 YAML mapping을 읽는다."""

        path = self.repo_root / relative_path
        if not path.exists():
            raise ValueError(f"Unknown YAML path: {relative_path}.")
        return self.load_yaml_mapping(path)

    def iter_yaml_files(self, relative_dir: str) -> tuple[Path, ...]:
        """repo root 기준 config group의 YAML 파일을 안정적인 순서로 반환한다."""

        root = self.repo_root / relative_dir
        return tuple(sorted(root.glob("*.yaml")))

    def load_yaml_mapping(self, path: Path) -> dict[str, object]:
        """YAML 파일을 mapping으로 읽고 catalog가 다룰 수 있는 shape로 제한한다."""

        raw = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
        if raw is None:
            return {}
        if not isinstance(raw, Mapping):
            raise ValueError(f"Expected mapping config at {path}.")
        return dict(raw)

    def resolve_script_path(self, job_config_path: str) -> str:
        """Hydra job config 경로에서 대응되는 script entrypoint 경로를 해석한다."""

        if job_config_path.startswith("conf/entrypoints/central_classifier_seed/"):
            return job_config_path.replace(
                "conf/entrypoints/central_classifier_seed/",
                "scripts/experiments/",
            ).replace(".yaml", ".py")
        if job_config_path.startswith("conf/entrypoints/central_ssl_control/"):
            return job_config_path.replace(
                "conf/entrypoints/central_ssl_control/",
                "scripts/experiments/",
            ).replace(".yaml", ".py")
        if job_config_path.startswith("conf/entrypoints/fl_ssl/"):
            return job_config_path.replace(
                "conf/entrypoints/fl_ssl/",
                "scripts/experiments/",
            ).replace(".yaml", ".py")
        if job_config_path.startswith("conf/entrypoints/prototype_pack/"):
            return job_config_path.replace(
                "conf/entrypoints/prototype_pack/",
                "scripts/prototypes/",
            ).replace(".yaml", ".py")
        if job_config_path.startswith("conf/entrypoints/data_pipeline/"):
            return job_config_path.replace(
                "conf/entrypoints/data_pipeline/",
                "scripts/datasets/",
            ).replace(".yaml", ".py")
        if job_config_path.startswith("conf/entrypoints/prototype_analysis/"):
            return (
                job_config_path.replace(
                    "conf/entrypoints/prototype_analysis/",
                    "scripts/experiments/",
                )
                .replace("prototype_strategy.yaml", "prototype_strategy_experiment.py")
                .replace(".yaml", ".py")
            )
        raise ValueError(
            "Unsupported job config path for catalog script resolution: "
            f"{job_config_path}."
        )

    def source_of_truth_for_module(self, module_name: str) -> str:
        """Python module 이름을 repo 상대 source path로 변환한다."""

        spec = importlib.util.find_spec(module_name)
        if spec is None or spec.origin is None:
            return module_name
        return self.relative_repo_path(Path(spec.origin))

    def relative_repo_path(self, path: Path) -> str:
        """repo 내부 path면 상대 경로로, 외부 path면 절대 경로로 표현한다."""

        resolved = path.resolve()
        try:
            return str(resolved.relative_to(self.repo_root))
        except ValueError:
            return str(resolved)
