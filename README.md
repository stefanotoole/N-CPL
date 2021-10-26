# Novelty Guided Critical Path Learning

This repository is the official implementation of "Width-based Lookaheads with Learnt Base Policies and Heurisitics Over the Atari-2600 Benchmark" submitted to Neurips-21.

## Requirements

Expirements were run using python 3.6.9.

To install requirements:

```setup
pip install -r requirements.txt
```
The following may also be required:
`apt install python3-opencv`

## Run

To run training/evaluation use `RIW_atari_process` from the `scripts` directory:

```runScript
python scripts/RIW_atari_process.py
```
The `forParallelRuns` directory keeps track of the completed runs such that the run script can be restarted without rerunning completed trials. Note that if you wish to re-run trials just delete the contents within `forParallelRuns`.

See the `run_atari_process.py` script for parameter/run options.

See the jupyter notebook `notebooks/SMRF_RTDPvsRand.ipynb` for code to run RTDP vs Random policies for testing for SMRF domains.
## Get Results

See jupyter notebook `notebooks/get_results.ipynb`.
