# Bonsai's test suite

The suite checks Bonsai's outputs (trees, branch lengths, log-likelihoods, posteriors, the files written
and their formats) against values derived independently of Bonsai's code, so that a rewrite of
`bonsai/` that keeps the outputs passes and one that changes them fails. TESTING.md describes the
pre-existing numerical tests; this file describes the shared harness and the rules every test follows.

## Running

From the repository root, with one BLAS thread per process:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHONPATH=$PWD python -m pytest -q                      # everything under tests/
PYTHONPATH=$PWD python -m pytest -q tests/area_c         # one area
PYTHONPATH=$PWD python -m pytest -q -m "not mpi"         # skip the mpiexec runs
```

Two environments:

- **Bonsai** (`requirements-dev.txt`, plus `mpi4py` for the MPI tests): runs `tests/`. Tests that need
  the Bonsai-scout packages skip themselves (`h5py`, `tables`).
- **Bonsai-scout** (`requirements-dev_bonsai_scout.txt`, plus `playwright install --only-shell chromium`
  once): runs `tests/` (MPI tests skip, as it has no mpi4py) and the browser tests,
  `python -m pytest bonsai_scout/tests`, which stay outside `testpaths`.

Before a change goes in, the suite should also pass in two variants of the Bonsai environment: the same pins on
an older Python (3.12; the README installs 3.14), and without mpi4py (the MPI tests skip, and Bonsai runs as one
process, as the README describes for a failed mpi4py installation). With uv, from the repository root:

```bash
uv venv --python 3.12 ../env312 && uv pip install --python ../env312 -r requirements-dev.txt mpi4py
uv venv --python 3.14 ../env-nompi && uv pip install --python ../env-nompi -r requirements-dev.txt
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHONPATH=$PWD ../env312/bin/python -m pytest -q       # Python 3.12, with the MPI tests
PYTHONPATH=$PWD ../env-nompi/bin/python -m pytest -q     # no mpi4py: the tests marked mpi skip
```

The MPI tests run Bonsai at 2 and 4 ranks; they need 4 free cores at once.

Never `pip install -e` the repository: tests import Bonsai from the checkout they live in, and
tests/conftest.py stops the run if any Bonsai module comes from somewhere else. Bonsai's scripts run as
subprocesses with `PYTHONPATH` set to the checkout.

Environment variables the harness reads or sets:

| Variable | Effect |
|---|---|
| `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` | set to 1 for every Bonsai subprocess; set them for pytest too |
| `HWLOC_COMPONENTS` | set to `-gl,-cuda,-nvml,-opencl,-rsmi,-levelzero` unless already set. Open MPI's hwloc otherwise probes GPUs (CUDA, NVML, X11) at every MPI start, which took 18 s for a single process and 24 s for `mpiexec -n 2` on the GPU server this suite was written on, against 0.5 s without the probe. Bonsai has no GPU code. |
| `BONSAI_TEST_WORKERS` | number of concurrent Bonsai processes (default: half the free cores, at most 8; an MPI run counts one per rank) |
| `MPLBACKEND` | `Agg` |

`pytest.ini` turns on `-ra`, `--strict-markers`, `--import-mode=importlib` and a 300 s per-test timeout
(pytest-timeout). Markers: `mpi` (needs `mpiexec` and mpi4py, skipped otherwise) and `scout` (needs the
Bonsai-scout environment).

## Layout

```
tests/
  conftest.py              puts tests/ on sys.path, loads bonsai_testfixtures, checks the checkout
  bonsai_testtrees.py      newick parser and writer, clades, splits, Robinson-Foulds distance
  bonsai_testdata.py       synthetic data generator (known tree -> Sanity / features input)
  bonsai_testref.py        reference log-likelihood and posteriors (dense Gaussian), Tree builder
  bonsai_testrun.py        write_config, run_bonsai, output readers, folder names
  bonsai_testfixtures.py   run pool, run fixtures, bonsai_globals_reset, MPI skips
  test_harness.py          tests of the harness itself
  TARGETS.md               every function, option and YAML key in scope, ticked when tested
  area_a/ ... area_d/      tests grouped by the part of the repository they cover (TARGETS.md, Areas)
  test_*.py                pre-existing numerical tests (TESTING.md)
```

**One rule decides where a test goes:** every test of module `foo.py` lives in a file named exactly
`test_foo.py`, in the folder of the area that owns it (TARGETS.md lists the owner and the file for every
target). Running only the files named after the modules a change touches then runs every test of those
modules; a test of `bonsai_treeHelpers.py` placed in `test_trees.py` would not run when that module changes. Pipeline
tests of `bonsai/bonsai_main.py` are `tests/area_c/test_bonsai_main.py`. Two areas may both have a
`test_bonsai_dataprocessing.py`; `--import-mode=importlib` lets duplicate basenames collect.

The pre-existing files (`tests/test_two_leaf_branch_length.py` and the others in TESTING.md) predate the
rule; TARGETS.md maps each to the module it tests. They take about 2 s, so a selective run should always
include them.

## The harness

### Synthetic data (bonsai_testdata.py)

```python
from bonsai_testdata import DatasetSpec, simulate, make_dataset, balanced_newick, caterpillar_newick
spec = DatasetSpec(newick=balanced_newick(4), n_genes=60, n_noise_genes=20, noise_sd=0.1, seed=0,
                   n_duplicates=0, layout='sanity2')          # or 'sanity_legacy', 'features'
sim = make_dataset(spec, folder)                             # writes the input files and truth.json
sim.tree, sim.truth_newick, sim.cell_ids, sim.gene_ids, sim.signal_genes, sim.truth_splits()
means, variances = sim.bonsai_units(genes)                  # per-cell likelihood in Bonsai's rescaled units
```

The tree is any newick string with branch lengths (presets: `balanced_newick`, `caterpillar_newick`,
`MULTIFURCATING_NEWICK`). Features diffuse as Brownian motion at unit rate in Bonsai's rescaled units;
the measurement of gene g in a cell is `mu_g + sqrt(v_g) * (position + noise_sd * N(0,1))`. The Sanity
layouts store the posterior under Sanity's N(0, v_g) prior, which Bonsai divides out again
(`test_sanity_output_divides_back_to_the_likelihood`). Noise genes do not diffuse and fall below the
default zscore cutoff; `n_duplicates` gives the first leaves an exact twin (`<name>d`) on zero-length
edges. Same spec, same files.

### Running Bonsai (bonsai_testrun.py)

```python
write_config(yaml_path, dataset, data_folder, results_folder, **yaml_options)  # via create_config_file.py
run = run_bonsai(config, step='all', mpi_ranks=None, timeout=240, pickup_intermediate=None)
run.returncode, run.stdout, run.stderr, run.results_folder, run.final_folder(), run.check()
run_bonsai_steps(config, steps=('preprocess', 'core_calc', 'metadata'))
stage_folder('final', zscore_cutoff=1.0)   # 'final_bonsai_zscore1.0'; also 'preprocessed', 'greedy', ...
read_edge_info, read_vert_info, tree_from_folder, read_newick_file, read_metadata_json,
read_metadata_bonsai, read_posteriors, posteriors_by_name
```

`run_bonsai` writes `bonsai_<step>.log` (command, exit code, output) next to the config file. Use
absolute paths everywhere: Bonsai's scripts resolve relative paths against the repository root.

### End-to-end runs as session fixtures (bonsai_testfixtures.py)

```python
from bonsai_testfixtures import EXAMPLE, bonsai_run_fixture, bonsai_job_fixture, run_spec
my_run = bonsai_run_fixture('my_run', run_spec(DatasetSpec(...), nnn_n_randommoves=20, nnn_n_randomtrees=2))

def test_something(my_run):
    my_run.check()
    my_run.sim, my_run.data_folder, my_run.final_folder()
```

`run_spec(dataset, steps=('all',), mpi_ranks=None, timeout=240, **yaml_options)`; `dataset` is a
`DatasetSpec` or `EXAMPLE` (the bundled 64-cell example, read in place). Each fixture runs once per
session; all fixtures the selected tests need start together when the first test starts, on a pool sized
by the free cores. An MPI run takes one slot per rank; slots are handed out first come, first served, so a
run that needs four slots is not passed over by later one-slot runs. A test that takes its run with
`request.getfixturevalue(name)` gets it started early only when `name` is one of its parameter values
(`@pytest.mark.parametrize('name', ['run_a', 'run_b'])`); a name built inside the test, such as
`'run_%d' % ranks`, starts when the test asks for it. For a custom sequence (run, delete a folder, run again with `pickup_intermediate`)
write `fn(pool, workdir)` and declare `bonsai_job_fixture(name, fn, slots=1, mpi=False)`; inside `fn`,
use `pool.dataset(spec)`, `write_config` and `run_bonsai`, never `pool.result()`. Fixture names are
global; reuse a name only for the same spec.

Cost: a 16-cell run with `nnn_n_randommoves=20, nnn_n_randomtrees=2` takes about 3 s, the 64-cell
example about 10 s; about 1 s of each is Python start-up and imports. The defaults (1000 moves x 10
trees) are much slower. The whole suite must stay near 3 minutes on a loaded server; reuse runs and
keep datasets small. The largest single costs are area B's greedy-merging runs at scale: about 9 s on one core
(200 cells), 11 s on four (200 cells, `-n 4`) and 29 s on four (520 cells through the memory-mapped exchange).

### In-process tests of Tree and TreeNode

`bonsai_globals_reset` (autouse) snapshots and restores `bonsai.bonsai_globals`, `mpi_wrapper.comm`,
numpy's global random state and the recursion limit around every test. Never import or run
`bonsai_main.py` or `create_config_file.py` in-process: both change the working directory and work at
import. `bonsai_testref.build_bonsai_tree(node_tree, means, variances)` builds a `Tree` from a newick
`Node` tree and per-leaf data and sets the globals Bonsai's tree code reads.

### Reference computations (bonsai_testref.py)

`reference_loglik(tree, means, variances)` is the dense multivariate-normal likelihood of the leaves
with a flat prior on the root; `reference_posteriors` gives every named node's posterior mean and
variance. Bonsai reports log-likelihoods in its own units:

```
calcLogLComplete = 2 * log p + (C - 1) * G * log(2 pi)  [+ loglikVarCorr = -(C - 1) * sum_g log v_g]
```

(`reference_bonsai_loglik` returns that number). Checked to 1e-15 relative against `calcLogLComplete`
on hand-built trees with polytomies and zero-length edges, and the posteriors against
`getAIRootInfo`; on a full run the final `finalLoglik` (metadata_bonsai.txt, no `loglikVarCorr`) and the
posterior files match the reference on the final tree (use `tree.nwk`, which keeps full precision;
edgeInfo.txt rounds lengths to `%.8e`).

### Topology (bonsai_testtrees.py)

`rf_distance(a, b)` compares unrooted topologies (non-trivial splits); `clades(tree)` compares rooted
ones. Both collapse edges of length `<= min_length` (default 0.0, so exact zero-length edges do not
count; pass `min_length=-1` to keep them). Bonsai roots its trees at the minimum-distance edge and
resolves polytomies into short edges, so compare against a multifurcating truth with `min_length` or
with "every true split is present".

## Rules for expected values

These bind every test in the suite.

- Derive expected values independently wherever practical: inputs small enough to work by hand, a
  short reference implementation inside the test (dense Gaussian likelihoods, closed-form branch-length
  optima, brute-force enumeration of all topologies for 4-5 leaves, numerical derivatives), the
  generator's recorded truth, the paper's model, the README's file formats. A test that only records
  the current output says so in its docstring ("pins the current output"); keep these few.
- No test of behaviour Bonsai gets wrong. Report it as a bug with a reproducer instead (see "Known
  bugs" below).
- No dependence on platform, Python version or MPI rank count unless the test is explicitly about
  rank-count invariance and Bonsai holds it. Compare floats with tolerances (`rtol=1e-9` on
  log-likelihoods and branch lengths), never byte for byte against stored values; compare topology, IDs,
  column layouts and file names exactly. No `ru_maxrss` units, no path separators.
- Do not pin implementation details a speed optimization could legitimately change while keeping
  Bonsai's outputs identical: the order of internal calls, log messages and timings, intermediate data
  structures, which pairs the UB bound skips, memory use. Pin outputs: the tree, branch lengths,
  log-likelihoods, posteriors, the files written and their formats, the intermediate folder names
  `--pickup_intermediate` relies on.
- Test data: synthetic by default; the repository's own `examples/` data may be read in place. No
  outside data.

## What Bonsai does that tests must allow for

- **Exit codes.** Input errors (MARG output, missing files, too old Sanity, bad cell IDs) end with a
  bare `exit()`, so the process exits 0. Check the log message and the missing outputs, not only the
  exit code. `exit("message")` calls exit 1.
- **Logging.** `mp_print` logs through `logging` to stderr; stdout is nearly empty. Use `run.log`.
- **metadata.json** stores absolute paths (`pathToOrigData`, `results_folder`, `processedDatafolder`);
  metadata_bonsai.txt stores `reconsTime` (seconds) and `coresUsed`.
- **Names.** Bonsai renames any node whose ID starts with `internal_`; cell IDs may not contain
  `: , ( ) ;` or spaces. The generator refuses such leaf names.
- **Cells at internal positions.** A cell can end on a zero-length edge under an internal node
  (`M0:0.0`).
- **Run-to-run.** Serial runs of the same config gave byte-identical outputs across three
  `PYTHONHASHSEED` values, on the 16-cell synthetic set and on the 64-cell example (except the absolute
  paths in metadata.json).
- **MPI rank count.** `nnnReorderRandom` raises `nnn_n_randomtrees` to the number of ranks when it is
  smaller, and the random trees' seeds depend on that number, so runs with fewer random trees than ranks
  differ between rank counts. With `nnn_n_randomtrees >= ranks`: on the 16-cell synthetic set, serial,
  `-n 2` and `-n 4` gave byte-identical files (about 100 runs, many concurrent); on the 64-cell example `-n 2` and `-n 4`
  were byte-identical to each other but not to serial: the greedy merge already ends with a different
  rooted tree, the final tree has the same rooted clades, branch lengths differ by up to 1.2e-3 and
  `finalLoglik` by 2.7e-3 of 65746 (relative 4e-8). On 200 synthetic cells (eight clades, 60 genes) the greedy
  tree on 4 ranks is identical to the serial one; on 520 cells it has the same unrooted topology but a different
  root, with log-likelihoods 4e-7 relative apart. Rank-count tests belong to area C and must use data
  where Bonsai holds the property. Once, a `-n 4` run of the 16-cell set inside a full-suite run stopped
  after the SPR step without writing a final folder; it did not recur in about 235 later runs, and its
  log was not kept. Every run now keeps `bonsai_<step>.log` next to its config, under pytest's basetemp.
- **Side effects.** `bonsai_main.py` creates an empty `<repository>/../data/<dataset>` folder (see
  "Known bugs"). Every config the harness writes uses the dataset name `bonsai_test`,
  and the session removes `../data/bonsai_test` at the end if it created it and it is still empty.
  Nothing is written into the checkout.

## Known bugs

Behaviour that contradicts the README, TESTING.md, a docstring, a YAML comment or the Bonsai paper is
reported as a bug, with a reproducer, instead of being tested. Found while writing the harness:

- `bonsai_main.py` creates an empty `<repository>/../data/<dataset>` folder on every run
  (`SCData(onlyObject=True)` makes `data_path()`, which falls back to `../data/<dataset>`).
- `Tree.from_newick` raises `TypeError` on every input (`max(-inf, None)` in `get_max_node_ind`).
- `zscore_cutoff`: the documented z_g subtracts the gene's mean and is compared with the cutoff; the
  code skips the mean and compares with the cutoff squared, so the two disagree for any cutoff but 1.
- A YAML file without `tmp_folder` crashes `bonsai_main.py` (`Run_Configs` defaults it to `None`), although
  the README calls every key but five optional.

## Coverage

Line coverage of the five packages (coverage.py 7.17, branches off, Python 3.14 on Linux), in an environment with
every dependency: `requirements-dev_bonsai_scout.txt` plus mpi4py, so the MPI tests run too. Subprocesses count:
Bonsai's scripts, every `mpiexec` rank and the Bonsai-scout app that the browser tests start. The pytest process
runs with `dynamic_context = test_function`; subprocesses start coverage from `COVERAGE_PROCESS_START`, pointing
at a configuration of their own with `core = sysmon` (under the default tracer the app answers too slowly for one
of the two browser tests) and `sigterm = true` (the browser tests stop the app with SIGTERM). "Existing tests" are the repository's own tests (`tests/test_*.py` and
`bonsai_scout/tests`) at b16ef6f, run in a worktree of that commit; the last column is those tests plus this
suite on this branch. Test files and `bonsai_scout/tests` are left out of the counts.

| Package | Lines | Existing tests (b16ef6f) | Existing tests and this suite |
|---|---:|---:|---:|
| `backbone_based_bonsai/` | 338 | 0 (0.0%) | 328 (97.0%) |
| `bonsai/` | 6,069 | 3,360 (55.4%) | 5,479 (90.3%) |
| `bonsai_scout/` | 4,159 | 2,281 (54.8%) | 3,359 (80.8%) |
| `downstream_analyses/` | 1,778 | 625 (35.2%) | 1,663 (93.5%) |
| `optional_preprocessing/` | 350 | 0 (0.0%) | 347 (99.1%) |
| `optional_preprocessing/batch_correction/` | 241 | 0 (0.0%) | 236 (97.9%) |
| total | 12,935 | 6,266 (48.4%) | 11,412 (88.2%) |

Per-test attribution counts only lines run in the pytest process; the Bonsai runs of the end-to-end tests happen in
subprocesses and are counted for the package, not for a test.

Not exercised: `mem_friendly` loading above 25,000 cells; the greedy-merge checkpoints written every 1,000 merges
and the NNI checkpoints every 100 moves during a run (only reading them back is tested); the Bonsai-scout `server`
function in-process (only through the browser tests); the rows of TARGETS.md left unticked.

92 of 497 public names are named by no test file (`tests_pr_tables.py names`); most run inside the scripts and
methods that call them:

- backbone_based_bonsai/backbone_based_bonsai.py: `get_command_config_file`
- bonsai/bonsai_dataprocessing.py: `Metadata.from_dict`, `Metadata.from_json`, `SCData.find_branching_points`, `SCData.read_in_data`, `SCData.throw_out_zeros`, `load_data_for_tree`
- bonsai/bonsai_helpers.py: `hierarchy_to_newick`, `pause_anim`
- bonsai/bonsai_treeHelpers.py: `TreeNode.copy_node_topology`, `TreeNode.find_midpoint_pair`, `TreeNode.getCenterLeaf`, `TreeNode.getChildInfo`, `TreeNode.getDerivativesDownstream`, `TreeNode.getLogLGradLambda`, `TreeNode.getLogLGradLambdaSingleGene`, `TreeNode.getNewPairs`, `TreeNode.getTopology`, `TreeNode.getTopologyWithTimes`, `TreeNode.get_edge_dict_node`, `TreeNode.get_new_nn_pairs`, `TreeNode.get_vert_ind_to_node_node`, `TreeNode.initializeMergeGeneral`, `TreeNode.initializeMergeRound`, `TreeNode.mergeChildrenRecursiveSingle`, `TreeNode.mergeDownstreamChildren`, `TreeNode.mergeDownstreamChildrenSingle`, `TreeNode.receiveMergeChildrenInfo`, `TreeNode.reset_root_node`, `TreeNode.sendMergeChildrenInfo`, `TreeNode.store_coords`, `TreeNode.store_coords_posterior`, `TreeNode.store_coords_posterior_memfriendly`, `TreeNode.to_newick_node`, `communicateMaxs`, `getNewUBInfo`, `initializeTask`, `optimiseLambdaStar`, `updateNNInfo`
- bonsai/mpi_wrapper.py: `GatherNpUnknownSize`
- bonsai_scout/bonsai_scout_app.py: `server`
- bonsai_scout/bonsai_scout_helpers.py: `Bonvis_figure.get_bg_collection`, `Bonvis_figure.get_bonvis_data`, `Bonvis_figure.get_cell_inds_subtree`, `Bonvis_figure.get_correct_cell_to_celltype`, `Bonvis_figure.get_edge_collection`, `Bonvis_figure.get_node_collection`, `Bonvis_figure.get_size_info`, `Bonvis_figure.plot_multi_obj_verts`, `Bonvis_figure.rearrange_branches`, `Bonvis_figure.reset_ed_layout`, `Bonvis_figure.tweak_dendro_layout`, `Bonvis_figure.tweak_ea_layout`, `Bonvis_figure.tweak_ed_layout`, `Bonvis_figure.update_bg`, `Bonvis_figure.update_colors`, `Bonvis_figure.update_fig_coords`, `Bonvis_settings.from_dict`, `Bonvis_settings.from_json`, `Coords_info`, `Coords_info.add_layout_details_to_tree`, `Coords_info.bg_data_to_ax`, `Coords_info.change_ly_style`, `Coords_info.get_bg_coords`, `Coords_info.get_coords`, `Coords_info.reconstruct_tree`, `Coords_info.set_coords`, `Coords_info.transform_coords`, `Coords_info.update_nn_index`, `Transf_info.get_zoom_origin`, `merge_cells_at_zero_dist`, `merge_cells_at_zero_dist_node`, `set_transform_ellipse_collection`
- bonsai_scout/my_tree_layout.py: `Layout_Tree.equalDaylightAll`, `Layout_TreeNode`, `Layout_TreeNode.getDsLeafs`, `Layout_TreeNode.get_ed_angles`, `Layout_TreeNode.get_edge_dict_node`, `Layout_TreeNode.get_vert_ind_to_node_node`, `Layout_TreeNode.positionChildren`, `Layout_TreeNode.reset_root_node`, `Layout_TreeNode.set_dendrogram_coords`, `Layout_TreeNode.set_eq_angles`, `copy_from_scdata_tree`
- downstream_analyses/get_cluster_helpers.py: `Cluster_TreeNode.find_midpoint_pair`, `Cluster_TreeNode.get_ds_and_parent_info_plus_dists`, `Cluster_TreeNode.get_vert_ind_to_node_node`, `Cluster_TreeNode.reset_root_node`, `Cluster_TreeNode.store_us_dists`, `Cluster_TreeNode.to_newick_node`
- downstream_analyses/get_clusters_max_diameter.py: `get_annotation_based_clustering_random`, `get_min_pdists_clustering_deprecated`
