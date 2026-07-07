import os
import subprocess
import sys
from pathlib import Path

import pytest
from shiny.pytest import create_app_fixture

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DATA_FOLDER = REPO_ROOT / "examples" / "example_data" / "simulated_binary_6_gens_samplingNoise"


@pytest.fixture(scope="session")
def example_results_folder(tmp_path_factory):
    """Build the tiny 64-cell example dataset fresh (same commands as the
    README's "Example 1") and point BONSAI_DATA_PATH/BONSAI_SETTINGS_PATH at
    it, mirroring the env-var contract run_bonsai_scout_app.py sets up before
    launching the app.

    Must be requested before `app` in every test's parameter list: pytest
    resolves same-scope sibling fixtures in parameter-list order, and the
    `app` subprocess needs these env vars set before it starts.
    """
    base = tmp_path_factory.mktemp("bonsai_scout_e2e")
    results_folder = base / "results"
    config_path = base / "example_configs.yaml"

    subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "bonsai" / "create_config_file.py"),
            "--new_yaml_path", str(config_path),
            "--dataset", "simulated_binary_6_gens_samplingNoise",
            "--data_folder", str(EXAMPLE_DATA_FOLDER),
            "--results_folder", str(results_folder),
            "--input_is_sanity_output", "True",
            "--zscore_cutoff", "1.0",
            "--nnn_n_randommoves", "10",
            "--nnn_n_randomtrees", "2",
            "--use_knn", "10",
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "bonsai" / "bonsai_main.py"),
            "--config_filepath", str(config_path),
            "--step", "all",
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "bonsai_scout" / "bonsai_scout_preprocess.py"),
            "--results_folder", str(results_folder),
            "--annotation_path", str(EXAMPLE_DATA_FOLDER / "annotation"),
            "--take_all_genes", "False",
            "--config_filepath", "",
        ],
        cwd=REPO_ROOT,
        check=True,
    )

    os.environ["BONSAI_DATA_PATH"] = str(results_folder / "bonsai_vis_data.hdf")
    os.environ["BONSAI_SETTINGS_PATH"] = str(results_folder / "bonsai_vis_settings.json")
    return results_folder


# An absolute Path (rather than a string) skips shiny.pytest's `request.path`-relative
# resolution, which only works at module/class/function scope, not session scope.
# The variable name `app` must match the `app` parameter name test functions use.
app = create_app_fixture(REPO_ROOT / "bonsai_scout" / "app.py", scope="session")
