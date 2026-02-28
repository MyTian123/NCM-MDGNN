# NCM-MDGNN: A Mutli-Dimensional Graph Differential Neural Network, Towards the future of smart materials science 
<img width="416" height="197" alt="image" src="https://github.com/user-attachments/assets/89bf4b2b-b2ac-465b-abbf-d715ef92069e" />



## Overview
We present Mutli-Dimensional Graph Differential Neural Network (MDGDNN).The MDGDNN model consists of data preprocessing, icos multi-level spherical graph construction and differential graph neural operators. The design patterns and data flow changes of each link are shown in Figure 1. Specifically, the data preprocessing process involves the fusion of small sample data, the multi-ML ensemble algorithms, and the Kalman filter denoising algorithm to achieve precise sample enhancement and data preprocessing. Secondly, a vectorization method is employed to construct a multi-stage subdivided spherical structure from the 0th level icosphere spherical surface. Meanwhile, we construct a three-dimensional spherical structure network with different level in a nested structure. The graph relationship matrix (including the vertex matrix and edge matrix) is recorded based on the relationship between vertices and edges. The CSR sparse storage structure is used to achieve efficient and lightweight construction of large-scale graph structures. For the core part of the model, we designed a graph convolution kernel (GCN kernel) based on multi-order differential graph operator (GDO), and integrated it with graph convolution networks and residual networks to form Graph Differential Operator Unit (GDO Unit). Furthermore, we have designed a Multi-scale Graph Aggregation Algorithm, called GDO-Unet, which continuously expands the influence range of the operator in the space through a graph U-net structure, in order to simultaneously model both local and global physical and chemical spatial patterns.

## Prepare
Our code requires 

```bash
torch, torch_geometric, scipy, pyigl
```
.

Please use pip to install these enviroments before training.

## Data acquisition and preprocessing
### Mesh Generation
Before operating the MDGNN model, users are required to generate the icosphere graph. This generation process is citing from [ugscnn](https://github.com/maxjiang93/ugscnn)
To acquire the mesh files used in this project, run the provided script gen_mesh.py.

```bash
python gen_mesh.py
```

To locally generate the mesh files, the Libigl library is required. Libigl is mainly used for computing the Laplacian and Derivative matrices that are stored in the pickle files. Alternatively, the script will download precomputed pickles if the library is not available.
However, we suggest the users using docker for mesh generation.

```bash
docker pull gilureta/pyigl
sudo nuhup docker run gilureta/pyigl >docker.txt 2>&1 &
```
Copy the code into the docker, and operation the gen_mesh.py. By this way the users will easily achieve the mesh.

### Data prepare
Before using the model, users should operate the data preprocess.
Enter the 'NCM-MDGNN/data_preprocess' dir, and use the preprocess.ipynb for preprocessing.

## Runing the model
For most experiments, simply running the script run.sh is sufficient to start the training process:

```bash
chmod +x run.sh
./run.sh
```
The script will automatically download data files if needed.




