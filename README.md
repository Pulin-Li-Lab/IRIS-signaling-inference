# IRIS-signaling-inference
code repository for Hutchins et al. "Reconstructing signaling history of single cells via statistical inference" (2025)


## Installation
[Create a Conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands) with Python=3.9, and activate the virtual environment:
```
conda create -n <my_env> python=3.9
conda activate <my_env>
```
Clone this repo:
```
git clone https://github.com/Pulin-Li-Lab/IRIS-signaling-inference.git
cd IRIS-signaling-inference
```
Install the other required packages:
```
pip install -r requirements.txt
conda install -c bioconda gseapy
```
