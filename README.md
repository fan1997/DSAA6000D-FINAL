# TC-GNN-test (Running Sparse GNN on Dense Tensor Core on Ampere GPU)

+ **This is the final project of DSAA6000D.** We test the [TC-GNN](https://arxiv.org/abs/2112.02052) on Nvidia Tesla A30 with some other datasets from Open Graph Library and DGL. The project is based on the original project: https://github.com/YukeWang96/TCGNN-Pytorch.
+ **Cite this project and [paper](https://arxiv.org/abs/2112.02052).**
```
@inproceedings{TC-GNN,
  title={TC-GNN: Accelerating Sparse Graph Neural Network Computation Via Dense Tensor Core on GPUs},
  author={Yuke Wang and Boyuan Feng and Yufei Ding},
  booktitle={Arxiv},
  year={2022}
}
```

+ **Clone this project**.
```
git clone git@github.com:YukeWang96/TCGNN-Pytorch.git
```

+ **OS & Compiler**: 
> + `Ubuntu 16.04+`
> + `gcc >= 7.5`
> + `cmake >= 3.14`
> + `CUDA >= 11.0` and `nvcc >= 11.0`

## Files and Directories.
+ `config.py`: the configuration file for the shape of a TC block.
+ `bench.py`: the benchmark file for invoking `main_tcgnn.py` for various datasets and models.
+ `main_tcgnn.py`: the main entry for running TC-GNN-test.
+ `TCGNN_conv/`: the directory for core TC-GNN-test implementations, including `TCGNN_kernel.cu` and `TCGNN.cpp`.

### [**Method**] Install via Conda.
+ Install **`conda`** on system **[Toturial](https://www.digitalocean.com/community/tutorials/how-to-install-anaconda-on-ubuntu-18-04-quickstart)**.
+ Create a **`conda`** environment: 
```
conda create -n env_name python=3.7.5
```
+ Install **`Pytorch`**: 
```
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge
```
or using `pip` [**Note that make sure the `pip` you use is the `pip` from current conda environment. You can check this by `which pip`**]
```
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
```
+ Install [**`Deep Graph Library (DGL)`**](https://github.com/dmlc/dgl)(optional).
```
conda install -c dglteam dgl-cuda11.0
pip install torch requests tqdm
```

+ Install [**`Pytorch-Geometric (PyG)`**](https://github.com/rusty1s/pytorch_geometric)(optional).
```
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
pip install torch-geometric
```

### Install **`TC-GNN-test`**.
Go to `TCGNN_conv/`, then run
```
./build.sh
```
to install the TCGNN_conv modules with Pytorch binding. 


### Download graph datasets.
Get the preprocessed datasets in `.npy` at [here](https://storage.googleapis.com/graph_dataset/tcgnn-ae-graphs.tar.gz), 
then run
```
tar -zxvf tcgnn-ae-graphs.tar.gz
```

If you wish to get datasets from DGL or OGB, 
run

```
python get_dgl_data.py
```

## Running **TC-GNN-test**.
> +  Under the current project directory 
> + `python main_tcgnn.py --dataset dataset_name --single_kernel_b_v1` to run the script and the report 10 epoch runtime for evaluated datasets. 
