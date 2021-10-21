# Ray-ONet: Efficient 3D Reconstruction From A Single RGB Image
**[Project Page](https://rayonet.active.vision/) | [Arxiv](http://arxiv.org/abs/2107.01899)**

Wenjing Bian, 
Zirui Wang, 
[Kejie Li](https://likojack.github.io/kejieli/#/home), 
[Victor Adrian Prisacariu](http://www.robots.ox.ac.uk/~victor/). BMVC 2021.

Active Vision Lab, University of Oxford.


## Table of Content
- [Environment](#Environment)
- [Demo](#Demo)
- [Dataset](#Dataset)
- [Usage](#Usage)
- [Acknowledgement](#Acknowledgement)
- [Citation](#citation)

## Environment
First you have to make sure that you have all dependencies in place.
The simplest way to do so, is to use [anaconda](https://www.anaconda.com/). 

You can create an anaconda environment called `rayonet` using
```
conda env create -f environment.yaml
conda activate rayonet
```

Next, compile the extension modules.
You can do this via
```
python setup.py build_ext --inplace
```

## Demo

You can now test our code on the provided input images in the `demo` folder.
To this end, simply run
```
python generate.py configs/demo.yaml
```
This script should create a folder `demo/generation` where the output meshes are stored.
The script will copy the inputs into the `demo/generation/inputs` folder and creates the meshes in the `demo/generation/meshes` folder.
Moreover, the script creates a `demo/generation/vis` folder where both inputs and outputs are copied together.

## Dataset

### Building the dataset
The dataset can be built according to the following steps:
* download the [ShapeNet dataset v1](https://www.shapenet.org/) and put into `data/external/ShapeNet`
* generate watertight meshes by following the instructions in the `external/mesh-fusion` folder. 
* download the [Preprocessed ShapeNet by Occupancy Networks](https://s3.eu-central-1.amazonaws.com/avg-projects/occupancy_networks/data/dataset_small_v1.1.zip) and put into `data/ShapeNet`.

You are now ready to build the dataset:
```
cd scripts
bash dataset_shapenet/build.sh
``` 

This command will build the dataset containing ground truth occupancies in `data/ShapeNet.build/ray_occ`.

## Usage
When you have installed all binary dependencies and obtained the preprocessed data, you are ready to train new models from scratch.
### Generation
To generate meshes using a trained model, use
```
python generate.py CONFIG.yaml
```
where you replace `CONFIG.yaml` with the correct config file.

The easiest way is to use a pretrained model.
You can download the pretrained model from this [link](https://drive.google.com/file/d/13bqvif27cNJJ5SzNtM4quttLoEy08pfH/view?usp=sharing) 
and place it under the directory `rayonet`

### Evaluation
For evaluation of the models, we provide two scripts: `eval.py` and `eval_meshes.py`.

The main evaluation script is `eval_meshes.py`.
You can run it using
```
python eval_meshes.py CONFIG.yaml
```
The script takes the meshes generated in the previous step and evaluates them using a standardized protocol.
The output will be written to `.pkl`/`.csv` files in the corresponding generation folder which can be processed using [pandas](https://pandas.pydata.org/).

For a quick evaluation, you can also run
```
python eval.py CONFIG.yaml
```
This script will run a fast method specific evaluation to obtain some basic quantities that can be easily computed without extracting the meshes.
This evaluation will also be conducted automatically on the validation set during training.

All results reported in the paper were obtained using the `eval_meshes.py` script.

### Training
To train Ray-ONet from scratch, run
```
python train.py CONFIG.yaml
```
where you replace `CONFIG.yaml` with the name of the configuration file you want to use.

You can monitor on <http://localhost:6006> the training process using [tensorboard](https://www.tensorflow.org/guide/summaries_and_tensorboard):
```
cd OUTPUT_DIR
tensorboard --logdir ./logs --port 6006
```
where you replace `OUTPUT_DIR` with the respective output directory.

For available training options, please take a look at `configs/default.yaml`.
## Acknowledgement
We thank [Theo Costain](https://www.robots.ox.ac.uk/~costain/) for helpful discussions and comments. We thank [Stefan Popov](https://www.popov.im/) for providing the code for 
[CoReNet](https://arxiv.org/abs/2004.12989) and guidance on training. Wenjing Bian is supported by China Scholarship Council (CSC).
 
Our code is built on [Occupancy Networks](https://github.com/autonomousvision/occupancy_networks). We thank the excellent code they provide.

## Citation
```
 @inproceedings{bian2021rayonet,
	title={Ray-ONet: Efficient 3D Reconstruction From A Single RGB Image}, 
	author={Wenjing Bian and Zirui Wang and Kejie Li and Victor Adrian Prisacariu},
	booktitle={BMVC},
	year={2021}
   }
```