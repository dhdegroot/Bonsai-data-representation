# *Bonsai*-data-representation
Algorithm to create an accurate and interpretable data-representation by reconstructing the maximum likelihood cell-tree from single-cell data, and visualizing this using the *Bonsai-scout* app.

## Input
The *Bonsai* tree-reconstruction was originally designed for the analysis of single-cell RNA-sequencing data, but can be used on any high-dimensional dataset that contains objects with with high-dimensional feature vectors. *Bonsai* needs to be provided with a vector of the most likely feature-values for each object, and takes into account standard-deviations on these feature estimates if provided, see the *Bonsai*-publication for details.

### *Bonsai* for scRNA-seq data using *Sanity*
When using *Bonsai* for scRNAseq-data, we highly recommend using *Sanity* for processing of the raw counts [Sanity's GitHub-page](https://github.com/jmbreda/Sanity), or to upload your data to our [single-cell pipeline](https://bonsai.unibas.ch). If running *Bonsai* after *Sanity*, *Bonsai* just requires the files in the output-directory created by *Sanity*, and can be run with the argument `--input_is_sanity_output True`. 
**IMPORTANT: Make sure to run *Sanity* with the flags `-e 1 -max_v true`**. This makes sure that we get *Sanity*'s extended output, and that it contains the inferred gene expression values corresponding to the maximum posterior guess of the gene-variances. See the SI of the *Bonsai*-publication for an extensive discussion of why this is the preferred mode of running *Sanity* when getting *Bonsai* input.

Example Sanity-command:
```
Sanity -n <number_of_threads> -f <path_to_gene_table.tsv> -d <path_to_output_folder> -e true --max_v true
```
or with an mtx-file:
```
Sanity -n <number_of_threads> -f <path_to_gene_table.mtx> -d <path_to_output_folder> -e true --max_v true -mtx_genes <path_to_gene_ids.tsv> -mtx_cells <path_to_cell_ids.tsv>
```

### *Bonsai* on general datatypes
If running *Bonsai* based on other data, read the Section on [Running *Bonsai* on other data-types](https://github.com/dhdegroot/Bonsai-data-representation#running-bonsai-on-other-data-types).

## Installing *Bonsai*
1. Clone the GitHub repository:
```
git clone https://github.com/dhdegroot/Bonsai-data-representation.git
```

2. Create a conda environment:
```
conda create --name bonsai python=3.14 -y
conda activate bonsai
```

3. Install mpi4py
For bigger datasets (~2000 cells and larger), we recommend running *Bonsai* using parallel computation. In that case, one needs to install *mpi4py* and therefore an underlying installation of *OpenMPI*. 
For running *Bonsai* on a computing cluster, see the below information on [Installing *mpi4py* on an HPC Cluster](https://github.com/dhdegroot/Bonsai-data-representation/blob/main/README.md#installing-openmpi-on-an-hpc-cluster). Running parallel computation on your personal computer will only lead to modest speed-up, but documentation for installing mpi4py can be found at [mpi4py GitHub](https://github.com/mpi4py/mpi4py/blob/master/INSTALL.rst).
Important note: Even with a failed installation of *mpi4py*, *Bonsai* will run, but it will only make use of 1 computing core.

4. Navigate to the cloned GitHub repository:
```
cd <PathToLocalBonsaiRepository>
```

5. Install the required *Python* packages by
```
pip install -r requirements.txt
```

### Installing *OpenMPI* on an HPC Cluster
We here follow the steps as described on [Installing OpenMPI on HPC Clusters](https://researchcomputing.princeton.edu/support/knowledge-base/mpi4py):

* Load the MPI version you want to use, we recommend using a relatively new version of OpenMPI. In order to get a list of all the available Open MPI versions on the cluster, run:
```
module avail openmpi
```
* Replace the version numbers in the line below if this version is not available, and then run the command:
```
module purge
module load OpenMPI/4.1.5-GCC-12.3.0
```
(When you eventually will start a *Bonsai* run, you will need to do this `module load` command again (for example in a *Slurm*-script). You will there also need to replace <x.y.z> by the correct version number.
* Set the loaded version of MPI to be used with mpi4py:
```
export MPICC=$(which mpicc)
```
Finally, we install mpi4py using `pip`:
```
pip install mpi4py==4.0.0 --no-cache-dir
```
It could be that you received the "Could not build wheels for mpi4py"-error. In that case, you may follow the steps below to complete installation by swapping which version of `ld` `pip` will use:
```
cd /home/<NetID>/.conda/envs/fast-mpi4py/compiler_compat
rm -f ld
ln -s /usr/bin/ld ld
```
Now you can rerun the pip install command above and it should succeed. Afterwards, you should point "ld" in your conda environment back to what it was:
```
cd /home/<NetID>/.conda/envs/fast-mpi4py/compiler_compat
rm -f ld
ln -s ../bin/x86_64-conda-linux-gnu-ld ld
```
Regardless of which version of Python you're using, you can check that mpi4py was properly installed by running:
```
python -c "import mpi4py"
```
If this command does not give an error-message, *mpi4py* was successfully installed.

## Running *Bonsai*
In this section, we will go over the steps to recontruct a Bonsai data-representation. These steps are also followed in the example below [(Example 1: Your first *Bonsai* run)](https://github.com/dhdegroot/Bonsai-data-representation?tab=readme-ov-file#example-1-your-first-bonsai-run), so as a good test if *Bonsai* and its dependencies were properly installed, you can follow that example.

The *Bonsai*-algorithm is started by calling the `bonsai_main.py`-script from the command line. As an argument, 
it needs a path to a YAML-file with run configurations, which you need to modify for your specific run first. 
### Preparing the configuration-file
The easiest way to create the configurations file is to use the provided script that can be found in 
`Bonsai-data-representation/bonsai/create_config_file.py`. One can for example run it by specifying only the path to where
you want to store this YAML-file. From the `Bonsai-data-representation` directory, run:
```
python3 bonsai/create_config_file.py --new_yaml_path <YOUR_DESIRED_CONFIG_FOLDER>/new_yaml.yaml
```
This will create a file at the specified path in which one can change the desired configurations with any text editor. 
In the YAML-file, one can find extensive explanations of the different run configurations.

Alternatively, one can give the desired configurations as arguments to the `create_config_file`-script. 
From the main directory "Bonsai-data-representation" one can for example run (note that all paths should be either absolute or 
relative to the "Bonsai-data-representation"-folder):
```
python3 bonsai/create_config_file.py \
  --new_yaml_path examples/1_simple_example/example_configs.yaml \
  --dataset simulated_binary_6_gens_samplingNoise \
  --data_folder examples/example_data/simulated_binary_6_gens_samplingNoise/ \
  --verbose True \
  --results_folder examples/1_simple_example/results/simulated_binary_6_gens_samplingNoise/ \
  --input_is_sanity_output True \
  --zscore_cutoff 1.0 \
  --UB_ellipsoid_size 1.0 \
  --nnn_n_randommoves 1000 \
  --nnn_n_randomtrees 2 \
  --pickup_intermediate False \
  --use_knn 10
```
*See the Section [Possible run configurations](https://github.com/dhdegroot/Bonsai-data-representation/blob/main/README.md#possible-run-configurations) for information on the different configurations that you can change. **For first usage, one only needs to customize the configurations `dataset`, `data_folder`, `filenames_data`, `results_folder`, `input_is_sanity_output`. The other configurations are only required for advanced usage.***

### Running Bonsai on a single core
Once you have prepared the YAML-file, you only need to start the *Bonsai* run. 
It now takes only two arguments. The argument `--config_filepath` is strictly required and should contain the path to the configuration-file (either relative to the "Bonsai-data-representation"-directory, or absolute. The `--step`-argument is optional, and allows us to cut the calculation up into bricks that have different computational resource requirements, see [Example 2](https://github.com/dhdegroot/Bonsai-data-representation/blob/main/README.md#example-2-running-bonsai-on-an-hpc-cluster) below, for an example. Setting `--step all` will just run the full algorithm at once. From the "Bonsai-data-representation"-directory, one thus runs:
```
python3 bonsai/bonsai_main.py \
  --config_filepath <YOUR_DESIRED_CONFIG_FOLDER>/new_yaml.yaml \ 
  --step all
```

### Running *Bonsai* using parallel computation
For a parallel run, one uses the same arguments, except that we call the script through *OpenMPI* like:
```
mpiexec -n <NUMBER_OF_CORES> python -m mpi4py bonsai_main.py \
  --config_filepath <YOUR_DESIRED_CONFIG_FOLDER>/new_yaml.yaml \
  --step all
```

### Restarting *Bonsai* from an intermediate point
It may happen that a *Bonsai*-run gets aborted, e.g. for exceeding some time-limit. Fortunately, *Bonsai* stores intermediate results at regular intervals. Using the argument `--pickup_intermediate True`, we can restart *Bonsai* and it will automatically find the latest intermediate result and re-start its computation from there. [Note: This functionality has only been introduced on July 9, 2025, so make sure to first pull a recent version of the code]. A typical run command would thus be:
```
mpiexec -n <NUMBER_OF_CORES> python -m mpi4py bonsai_main.py \
  --config_filepath <YOUR_DESIRED_CONFIG_FOLDER>/new_yaml.yaml \
  --step all \
  --pickup_intermediate True
```

### The *Bonsai* results
The results will be stored in the results-folder as indicated in the .yaml-file. It will contain the following type of results:
* **a subdirectory starting with `final_bonsai_`.** This directory will contain the final state of the tree stored in two ways. 
  - First, there is a `.nwk`-file which is a universally accepted format for storing trees. 
  - Second, the files `edgeInfo.txt`, `vertInfo.txt`, `metadata.json` together contain a more human-readable form of storing the tree. These files are necessary for visualizing the tree using the `bonsai_scout`-app.
    - `edgeInfo.txt` contains three columns, and every row defines an edge in the tree. The first two columns indicate the `vertInd` (vertex-index) of the vertices connected by the edge, the third column indicates the length of the edge
    - `vertInfo.txt` contains for each vertex its `vertInd` and its `vertName`. The `vertName` is important for coupling the original single-cell data to the tree. The column `nodeInd` is a historical artefact, and its usage will be removed in the near future.
    - `metadata.json` contains some metadata on the dataset, importantly, it also contains a path to in what folder the original gene expression data can be found.
  - Third, the files `posterior_ltqs_vertByGene.npy` and `posterior_ltqsVars_vertByGene.npy` contain the inferred gene-expression values for all nodes, i.e. also the internal nodes, when we have marginalized over all other node positions. These values thus really indicate our best guess of the gene expression levels in that node. Given the reconstructed tree, these data can be always be reconstructed from the original data, but storing it here in a binary-format saves time when we want to visualize the tree with the data.
* several subdirectories starting with `intermediate_bonsai_`. In these folders, the state of the reconstructed tree is stored as it was after the different steps in the *Bonsai* reconstruction algorithm. These directories are very useful when one wants to run only part of the *Bonsai*-algorithm again from one of the intermediate steps, for example when the run failed due to running out of time or memory. Since the data-files `..._vertByGene.npy` are removed from these subdirectories, the memory footprint of these folders is small. Still, one may choose to remove these directories when the full *Bonsai* run has completed.
* a copy of the configuration YAML-file with an added timestamp. In this way, one can always see with what running configurations the results were created, and when this run was started.

## Running Bonsai on other data-types
As we describe in the *Bonsai*-paper, *Bonsai* can be used to reconstruct the maximum likelihood tree on any dataset with objects that have features in a high-dimensional feature space. However, it is necessary that the data is normalized first to align with the likelihood model that *Bonsai* uses. In short, *Bonsai* assumes that, for each object, the input specifies a vector of estimated features (`features.txt`) with error-bars (`standard_deviations.txt`). The input data should be normalized such that the likelihood of the `measured` features is reasonably approximated by a multi-variate Gaussian with means given by `features.txt` and standard-deviations given by `standard_deviations.txt`, and negligible covariances. As we extensively discuss int he paper for scRNA-seq data, this may require a careful definition of the variables to be inferred, which will depend on the data type. Apart from the specific application for scRNA-seq, it will be up to the user to provide appropriately normalized data to _Bonsai_.

### The required files
To use *Bonsai* on a general dataset that contains $C$ objects in an $G$-dimensional feature space, we need to provide it with:
* `features.txt`: This should be a tab-separated file containing a matrix with $G$ rows and $C$ columns. The file should only contain the data, no header or index. Column $i$ should give the best estimates of the feature values for object $i$. 
* (optional) `standard_deviations.txt`: This should be a tab-separated file containing a matrix with $G$ rows and $C$ columns. The file should only contain the data, no header or index. Column $i$ should give standard deviations corresponding to the measurement noise on the features, i.e., they should give the uncertainty on the feature values provided in `features.txt`. If `standard_deviations.txt` is not provided, *Bonsai* will assume that the uncertainty on the feature values is very small.
* (optional) `cellID.txt`: Here, one can provide IDs for the objects. This should be a simple text file with on each row an object-ID corresponding to the columns in `features.txt`. *Note that we extend the term `cellID` here to refer to all object-IDs, this is a harmless historical artifact.*
* (optional) `geneID.txt`: Here, one can provide IDs for the features. This should be a simple text file with on each row a feature-ID corresponding to the rows in `features.txt`. *Note that we extend the term `geneID` here to refer to all feature-IDs, this, again, is a harmless historical artifact.*

### The Bonsai run-configurations for general data types
After the data have been properly normalized, running *Bonsai* happens similarly to what was described before for *Sanity*-output. The major differences are:
* The argument `--input_is_sanity_output` should now be set to `False`.
* The argument `--filenames_data` should now be used to point *Bonsai* to the files with features and error-bars, while for *Sanity*-output we know that this is given by `delta_vmax.txt,d_delta_vmax.txt`. So, for example, one could give `--filenames_data features.txt,standard_deviations.txt`.

## *Bonsai-scout*: Visualizing the *Bonsai* results
The reconstructed tree can be visualized in the Bonsai-scout-app that was developed for this. 

### Installing the required packages
First, make sure that you have the required packages installed. Since, we need a few packages more than for the *Bonsai*-reconstruction and we wanted to keep the *Bonsai*-dependencies as lean as possible, we created a new conda environment for *Bonsai-scout*. (If you created this environment before, you can easily switch with `conda activate bonsai_scout`).

* Make a new conda environment:
```
conda create --name bonsai_scout python=3.14 -y
conda activate bonsai_scout
```
* Install the required packages:
```
pip install -r requirements_bonsai_scout.txt
```

### Preprocessing for the visualization
Before running the app, we must first do a preprocessing run, using (from the directory "Bonsai-data-representation")
```
python3 bonsai_scout/bonsai_scout_preprocess.py \
--results_folder <BONSAI_RESULTS_FOLDER> \
--annotation_path <PATH_TO_CELL_ANNOTATION_FOLDER> \
--take_all_genes False \
--config_filepath ''
```
You need to provide the results-folder where the *Bonsai*-results were stored. This directory normally also contains a copy of the configurations file that was used for *Bonsai*. If this is not the case, or not the correct YAML-file, then you can use the `--config_filepath` argument to override this. The argument `--take_all_genes` can be set to True if you want to be able to overlay the expression of all genes on the tree. By default, only the genes are available on which *Bonsai* reconstructed the tree (those with high enough signal-to-noise). Setting `--take_all_genes True` will increase memory usage and slow down the preprocessing.

In the folder under `--annotation_path`, one can provide several files in `.csv`- or `.tsv`-format that contain annotation for all the cells. These files should contain one column with the cell-IDs, and the next columns should all correspond to one annotation-type. These can be celltype-categories, but can also be numerical values. These annotations can be overlaid in the plot. An example-file is 
```
cellID,experimenter,measurement
55K12032T_S5,T,0.7
4230K12033A_S30,A,0.84
4028K12031A_S28,A,0.29
11K12011T_S1,T,0.493
```
One can also provide full feature matrices. These additional features can be visualized on the trees just like gene expression data, or can be used to call marker features for a specific subset on the tree. **Make sure to start the filename of this annotation-file with `mat_`, for example `mat_UMI_counts`.** The file should be in the same format as a normal annotation file, but the preprocessing will assume that only numerical values are given.

The preprocessing step will create two files and store them in the *Bonsai* results-folder:
- `bonsai_vis_data.hdf` which contains all the necessary data. This file will not be changed while running *Bonsai-scout*.
- `bonsai_vis_settings.json` which contains the default settings for the visualization. Since this `.json`-file is human-readable and -editable, one can edit some settings manually, for example for picking a customized colormap.

### Running *Bonsai-scout*
Now you can view your results in *Bonsai-scout* by running (from the "Bonsai-data-representation"-directory)
```
python3 bonsai_scout/run_bonsai_scout_app.py \
--results_folder <BONSAI_RESULTS_FOLDER> \
--settings_filename bonsai_vis_settings.json \
--port 1234
```
There should now be a print message pointing you to the correct link: 
```Your app will shortly be running at: http://0.0.0.0:1234. Use your browser (not Safari) to view it.```
and
```INFO:     Uvicorn running on http://0.0.0.0:1234 (Press CTRL+C to quit)```

**Warning: if you are running these commands through an SSH-connection, your link will not be correct. In that case you should do something like `http://<ADDRESS_OF_MY_SSH_NODE>:8243/`**, where 1234 matches the port that is reported to you in the print-messages, and that you could have picked yourself using the `--port` argument.

## Backbone-based *Bonsai*: for the largest datasets
For reconstructing *Bonsai*-trees on datasets of over a million cells, we developed a backbone-based version of *Bonsai*. It comprises the following steps:
1) Preprocessing of the data
2) A normal *Bonsai*-tree-reconstruction on a subset of `n_initial_cells` cells, which will be used as a backbone
3) Placing remaining cells until the backbone has grown by a certain factor (`growth_factor_guide`), or until all cells were added.
4) Refining the resulting tree using a normal *Bonsai*-run.

If not all cells were added, i.e. `n_initial_cells * growth_factor_guide` is smaller than the number of cells in the dataset, then we will use the result of Step 4 as a backbone, and continue at step 3.

### Parallellizing Backbone-based *Bonsai*
It is in principle possible to run backbone-based *Bonsai* on a single CPU which may be nice for testing purposes. In this case, one can set the argument `--return_commands False`, and just run `backbone_based_bonsai/backbone_based_bonsai.py` with the arguments-of-choice. 

However, this does not make sense for large datasets. In that case, one should first run `backbone_based_bonsai.py` **on a single CPU**, but set `--return_commands True`. In this case, a file will be created in the Bonsai-results folder that is called `iterative_build_commands.txt`. In this file, each line contains a run command that can be used in a normal MPI-call (similar as described in [Running Bonsai using parallel computation](https://github.com/dhdegroot/Bonsai-data-representation/blob/main/README.md#running-bonsai-using-parallel-computation)).
It will be helpful to know the rough computational requirements for the various steps:
1) Preprocessing -> limited time, significant memory requirement, benefits from parallellization
2) Backbone-tree -> limited time and memory requirements, depending on the number of initial cells in the backbone, benefits from parallelization.
3) Adding cells -> significant time and memory requirement, can use only a single CPU
4) Refining the resulting tree -> significant time and memory requirement, benefits from parallellization

## Example 1: Your first *Bonsai* run
To test if *Bonsai* is set-up correctly, let's run it on a very small dataset of 64 cells. The output from *Sanity* for this dataset is stored in `Bonsai-data-representation/examples/example_data/simulated_binary_6_gens_samplingNoise`. 
Let's create a YAML-file with the correct run configurations first. To keep things organized, we can store the YAML-file in the `1_simple_example`-folder:
```
python3 bonsai/create_config_file.py \
  --new_yaml_path examples/1_simple_example/example_configs.yaml \
  --dataset simulated_binary_6_gens_samplingNoise \
  --data_folder examples/example_data/simulated_binary_6_gens_samplingNoise/ \
  --verbose True \
  --results_folder examples/1_simple_example/results/simulated_binary_6_gens_samplingNoise/ \
  --input_is_sanity_output True \
  --zscore_cutoff 1.0 \
  --nnn_n_randommoves 1000 \
  --nnn_n_randomtrees 2 \
  --use_knn 10
```
You can open the YAML-file and see how the parameters are set. Also, please read the extensive comments to familiarize yourself with what the different options do.

We can now move to running *Bonsai*:
```
python3 bonsai/bonsai_main.py \
 --config_filepath examples/1_simple_example/example_configs.yaml \
 --step all
```
This can take a few minutes. At the moment there is still a lot of output printed, that gives an idea of the progress of *Bonsai. We will clean up this output in the future.

Results will be stored in the indicated results_folder: `examples/1_simple_example/results/simulated_binary_6_gens_samplingNoise/`. If the *Bonsai*-run fully completed, there will be a directory called `final_bonsai_zscore1.0`. You can view the results there.

We will now visualize the tree with some cell-annotation. For this, we already provide an annotation-folder at `examples/example_data/simulated_binary_6_gens_samplingNoise/annotation`. Since this simulated dataset is a binary tree, we annotated the cells by their clade at different heights of the tree. 

Now run the preprocessing of the visualization:
```
python3 bonsai_scout/bonsai_scout_preprocess.py \
  --results_folder examples/1_simple_example/results/simulated_binary_6_gens_samplingNoise/ \
  --annotation_path examples/example_data/simulated_binary_6_gens_samplingNoise/annotation \
  --take_all_genes False \
  --config_filepath ''
```
Finally, run the following command to start *Bonsai-scout*:
```
python3 bonsai_scout/run_bonsai_scout_app.py \
--results_folder examples/1_simple_example/results/simulated_binary_6_gens_samplingNoise/ \
--settings_filename bonsai_vis_settings.json \
--port 1234
```
There should now be a print message pointing you to the correct link: 
```Your app will shortly be running at: http://0.0.0.0:1234. Use your browser (not Safari) to view it.```

## Example 2: Running *Bonsai* on an HPC-cluster
We will here give an example script on how to run *Bonsai* on an HPC cluster. The example is based on a Slurm-environment, but the basic principles can probably be generalized to other HPC-architectures.

We have divided *Bonsai* into three main steps: `preprocess`, `core_calc`, and `metadata`. The steps `preprocess` and `metadata` read in the data on all genes, while `core_calc` only deals with the genes that have a signal-to-noise ratio higher than the parameter `zscore_cutoff` which we recommend to set to 1. Therefore, the `core_calc`-step typically needs less allocated memory than the other steps. In contrast, `core_calc` is the most time-consuming. Both `preprocess` and `core_calc` benefit strongly from parallelization over multiple computing cores. 

An example *slurm*-script is given in `examples/2_bonsai_on_slurm/example_slurm_script.sh`, and we also added it below in this read-me. Note that there are some constants that one may need to adapt depending on the size of your dataset, i.e. `N_CORES`, `TIME`, `QOS`, `MEM_PER_CORE`. Also, make sure to adapt `PATH_TO_CODE` to have the absolute path to the `Bonsai-data-representation`-directory.

The `for`-loop that currently runs over only one value, can be used to start *Bonsai* on several datasets (with run configurations defined in their YAML-files) with only one script. The paths to the YAML-files have to be stored on different lines in a file called `yaml_paths.txt` which should be in the same folder as the slurm-script. The slurm-script will call a bash-script that executes the final *Python* call and that should also be in the same folder. You can find our example bash-script below.

One can now start the Bonsai-runs on all the datasets comprised in `yaml_paths.txt` by calling `bash example_run_script.sh`.

**example_slurm_script.sh**
Note that in the following you still have to change the arguments to `<PATH_TO_BONSAI_DATA_REPRESENTATION_FOLDER>` and `<YOUR_EMAIL>`.
```
#!/bin/bash

mkdir -p outerr

module purge
module load OpenMPI/4.1.5-GCC-12.3.0

export MPICC=(which mpicc)

PATH_TO_CODE="<PATH_TO_BONSAI_DATA_REPRESENTATION_FOLDER>"
SCRIPT_PATH="bonsai_bash.sh"

ID="bonsai_example"
NCORES=5
TIME="00:30:00"
QOS=30min
MEM_PER_CORE=4

mem=$((NCORES*MEM_PER_CORE))G

file="yaml_paths.txt"
line_count=$(wc -l < "$file")

for (( i=1; i<=$line_count; i++ ))
do
  YAML_PATH=$(head -n ${i} ./yaml_paths.txt | tail -1)

  echo Starting Bonsai based on configuration file: ${YAML_PATH}

  preprocess_id=$(sbatch --export ALL  --parsable --mem=${mem} --cpus-per-task=1 --ntasks=${NCORES} --nodes=1 --time=00:30:00 \
  --output=./outerr/pre_dataset${i}%x_%j.out \
  --error=./outerr/pre_dataset${i}%x_%j.err --qos=30min \
  --job-name="${ID}"_${i} \
  ${SCRIPT_PATH} -n ${NCORES} -s preprocess -y ${YAML_PATH})

  #preprocess_id is a jobID; pass to next call as dependency
  #Start main tree reconstruction after preprocess_id=ok

 core_calc_id=$(sbatch --export ALL  --parsable --mem=${mem} --cpus-per-task=1 --ntasks=${NCORES} --nodes=1 --time=${TIME} \
  --dependency=afterok:$preprocess_id \
  --output=./outerr/core_dataset${i}%x_%j.out \
  --error=./outerr/core_dataset${i}%x_%j.err --qos=${QOS} \
  --job-name="${ID}"_${i} \
  --mail-type=END,FAIL \
  --mail-user=<YOUR_EMAIL> \
  ${SCRIPT_PATH} -n ${NCORES} -s core_calc -y ${YAML_PATH})

 sbatch --export ALL  --mem=${mem} --cpus-per-task=1 --ntasks=1 --time=00:30:00 \
  --dependency=afterok:$core_calc_id \
  --output=./outerr/metadata_dataset${i}%x_%j.out \
  --error=./outerr/metadata_dataset${i}%x_%j.err --qos=30min \
  --job-name="${ID}"_${i} \
  ${SCRIPT_PATH} -n ${NCORES} -s metadata -y ${YAML_PATH}

done
```

**bonsai_bash.sh**
In the following, you again have to replace `<PATH_TO_BONSAI_DATA_REPRESENTATION_FOLDER>`:
```
#!/bin/bash

while getopts n:s:y: flag
do
    case "${flag}" in
        n) NCORES=${OPTARG};;
        s) STEP=${OPTARG};;
        y) YAML_PATH=${OPTARG};;
    esac
done

PATH_TO_CODE="<PATH_TO_BONSAI_DATA_REPRESENTATION_FOLDER>"

srun python -m mpi4py ${PATH_TO_CODE}/bonsai/bonsai_main.py --config_filepath ${YAML_PATH} --step ${STEP}
```

## Detailed description of the run configurations
As described above, one can set *Bonsai*'s run configuration using the provided `bonsai/create_config_file.py` that uses a template YAML-config file. This script already contains extensive comments that explain the different configurations, but for completeness we also describe the different configurations here.

### Specifying the input
* `--dataset [str, required]`:
Identifier of dataset.
* `--data_folder [str, default='']`:
Path (absolute or relative to "bonsai-development") to folder where input data can be found. If `--input_is_sanity_output True`, this folder should be the output-folder of _Sanity_, otherwise, this folder should contain a file with means and standard-deviations in files pointed to by the argument `--filenames_data`.
* `--filenames_data [str, default=delta.txt,d_delta.txt]`:
Filenames of input-files for features and standard deviations separated by a comma. These files should have different cells as columns, and features (such as log transcription quotients) as rows.
* `--results_folder [str, default='']`:
Path (absolute or relative to "bonsai-development") to folder where results will be stored.
* `--input_is_sanity_output [bool, default=true]`:
The provided means and stds by *Sanity* are the means and stds of the posterior for the feature: P(x | D), but in Bonsai we need the probability of the data given the feature value: P(D | x). These are related through P(x | D) ~ P(D | x) P(x). So, we need to divide out the prior. If the input is Sanity-output we know how to do this because the reported posterior is Gaussian and the prior was Gaussian around zero (see the SI of the Bonsai-publication). If your input-data is not Sanity-output, set this flag to False, but make sure to provide *Bonsai* with the required means and standard-deviations of the likelihood (P(D | x)).

### Run parameters
* `--zscore_cutoff [float, default=1]`:
Genes with a signal-to-noise ratio under this cutoff will be dropped. Negative means: keep all. Due to the large number of genes dominated by noise, tree-reconstruction is usually better when we discard the most noisy genes, even though we account for error-bars rigorously. The signal-to-noise ratio is defined as: z_g = frac{1}{C} sum_c frac{(delta_{gc} - bar{delta_g})^2}{epsilon_{gc}^2} where delta_{gc} are the posterior-means for gene g and cell c, and epsilon_{gc} is the standard-deviation on the posterior. If `--input_is_sanity_output` is false, these posterior values are derived from the provided features and standard-deviations by _Bonsai_, otherwise we use the *Sanity* output directly.
* `--rescale_by_var [bool, default=True]`:
This determines whether coordinates are rescaled by the inferred variance per gene (feature). by default *Bonsai* uses a prior where the amount of diffusion in dimension `g` is assumed proportional to the variance, `v_g` in that dimension. However, if desired the user can specify to not scale the prior by the variances $v_g$.

### Optional preprocessing
* `--tmp_folder [str, default='']`: (ADVANCED)
Path (absolute path or relative to "bonsai-development") pointing to a tree-folder from which Bonsai reconstruction will start. This is relevant if one wants to start from a tree other than the usual star-tree, for example if one already created a tree-object where [cellstates](https://github.com/nimwegenLab/cellstates) were connected to a common ancestor. This can lead to significant speed-up and to better tree reconstructions. For using Cellstates-output to create this initial tree, we provide a script `Bonsai-data-representation/optional_preprocessing/create_cellstates_premerged_tree.py` that can take the same YAML-config file as *Bonsai* and *Cellstates*-output, and creates the desired tree in a folder. The path to the created folder can be used in this argument.

### Computational optimization parameters
* `--nnn_n_randomtrees [int, default=10]`: (ADVANCED)
Decides how many random trees we create before taking the tree with the highest loglikelihood and doing nearest-neighbor interchanges greedily. Since we parallelized this procedure over the number of random trees, it never hurts to set nnn_n_randomtrees equal to the number of cores that are anyhow reserved, since then the different random trees will be created in parallel. Increasing this number can lead to moderate increases in the likelihood of the final *Bonsai*-tree, but will come at significant computational cost.

* `--nnn_n_randommoves [int, default=1000]`: (ADVANCED)
Decides how many random nearest-neighbor-interchange-moves we do before doing them greedily. Increasing this number can lead to moderate increases in the likelihood of the final *Bonsai*-tree, but will come at significant computational cost.

* `--use_knn` [int, default=10]: (ADVANCED)
Decides whether nearest-neighbours are used to get candidate pairs to merge. Set to -1 for considering all pairs of leafs. Values between 5 and 20 give good results. Computation time will scale approximately linear with `--use_knn`, while tests show that the tree likelihood will increase only slightly with this parameter.

* `--UB_ellipsoid_size [float, default=1.0]`: (ADVANCED)
This is a purely computational optimization that does not affect the final result. Decides whether we calculate an upper bound (UB) for the increase in loglikelihood that merging a certain pair can yield, given that a the root stays in a certain ellipsoid. Using these upper bounds, the calculation can be sped up enormously. If the UB_ellipsoid_size < 0, this estimation will not be used. Note that this will not lead to a different final tree, so we recommend not using that setting. Otherwise, it decides how large the ellipsoid is in root-position/precision space for which we estimate the UB. Larger values will result in looser UB, so that more candidate pairs per merge have to be considered. However, it will also result in longer validity of the UB-estimation, so that more merges can be done without re-calculating the upper bounds. Ellipsoid sizes below 5 are reasonable to try, we recommend to start at 1. Ellipsoid size will be updated dynamically by Bonsai based on the results.

### Starting from the intermediate results of previous runs
If *Bonsai* was already ran with these configurations, but failed (e.g. because of a time limit), it can be re-started from the last completed results by setting --pickup_intermediate True. In that case, we will look for the furthest advanced result in the results-folder. Note that the results-folder thus has to be cleaned of results that you don't want to be picked up.

* `--pickup_intermediate [bool, default=false]`:
Decides whether we look for intermediate results from previous runs or not. These intermediate results are periodically stored during any normal run, and can thus be used when a run did not finalize.

## Optional downstream analyses
In *Bonsai-scout*, we, next to interactive visualization, offer 1) tree-based clustering, 2) marker gene/feature detection, and 3) calculating the average features for different groups. The necessary code is provided in the `downstream_analyses`-subfolder. 
* Tree-based clustering. One can use the function `get_min_pdists_clustering_from_nwk_str` in the file `get_clusters_max_diameter.py` to perform minimal-distance clustering based on a tree given by a Newick-string. These clusters are also provided in *Bonsai-scout* and can there be downloaded by clicking the correct button.
* Calculating marker genes. See the Methods-section of the *Bonsai*-publication for an extensive description of the marker gene calculation. The marker genes can be calculated in *Bonsai-scout*, but for large datasets this calculation can take too long. In that case, one can download the necessary information on the selected groups in a `.json`-file. This `.json`-file can be used to run `calc_marker_genes.py`.
* Averaging over groups. Given a group of objects and the posterior feature-values (including error-bars) that *Bonsai* has calculated, one can use `average_over_groups_wrapper.py` to calculate the group's mean feature value and the group variance. Note that this can deviate from the naive mean and variance, because this script takes into account the uncertainty on the inferred feature values.
