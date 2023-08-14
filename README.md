# EGNAS
EGNAS: Efficient Graph Neural Architecture Search through Evolutionary Algorithm

### Installation

- conda environment setting
```
conda create -n EGNAS python=3.10 -y
conda activate EGNAS
```
- requirements
```
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia -y
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
pip install torch_geometric
pip install TorchSnooper hyperopt
```
### Quick start

```
sh scripts/EGNAS_Cora.sh
sh scripts/EGNAS_Citeseer.sh
sh scripts/EGNAS_Pubmed.sh
```