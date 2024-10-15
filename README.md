## Acknowledgements

This repository is a fork of the [original repository](https://github.com/pratyushasharma/laser) created by [Original Author(s)](https://github.com/pratyushasharma).


This repository contains code adapted from the original work on **Layer-Selective Rank Reduction (LASER)**. Our modifications expand the intervention capabilities by introducing new modes of operation through an additional module called **TASER**.

**Note:** This repository is a work in progress, and future updates will be made to enhance functionality and performance.



## Additions

TASER introduces new ways of compressing weight matrices in a transformer architecture :

- **Mode 1**: QKVO interventions across the entire model.
- **Mode 2**: QKVO interventions layer by layer.
- **Mode 3**: QKVO segmented into 3 parts.
- **Mode 4**: Fully Connected (FC) layers across the model.
- **Mode 5**: FC layer by layer.
- **Mode 6**: FC layers segmented.

These modes provide additional flexibility for model interventions, enabling a broader exploration of model performance under different structural modifications.

## Code Organization

The main code is located in the `src` folder. You will find two subfolders:
- **laser/**: Contains the original LASER implementation.
- **taser/**: Contains our modified TASER implementation with the additional intervention modes described above.

The experiment scripts follow a similar naming convention as in the original repository, with the format `TASER_intervention_<llm-name>_<dataset-name>.py`, where:
- `<llm-name>` refers to the name of the language model.
- `<dataset-name>` refers to the dataset used.

## Running Experiments

To run experiments with TASER, simply specify the intervention mode and rank range in the script. 

python src/TASER_intervention_model_dataset.py


