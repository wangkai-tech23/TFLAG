# TFLAG
This repository contains the implementation of the approach proposed in the paper "***TFLAG: Towards Practical APT Detection via Deviation-Aware Learning on Temporal Provenance Graph***".

### Environment Configuration

```Bash
conda create -n TFLAG python=3.9
conda activate TFLAG
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
wget https://data.pyg.org/whl/torch-1.10.0%2Bcu113/torch_scatter-2.0.9-cp39-cp39-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-1.10.0%2Bcu113/torch_sparse-0.6.12-cp39-cp39-linux_x86_64.whl
pip install torch_scatter-2.0.9-cp39-cp39-linux_x86_64.whl
pip install torch_sparse-0.6.12-cp39-cp39-linux_x86_64.whl
pip install torch_geometric==2.0.4
conda install -c conda-forge scikit-learn
pip install numpy==1.23.5

```

Create two new folders:`time_windows/test_data` and `time_windows/test_data`, under the `dataset/` folder in advance.

#### Overall architecture

Run `train.py` to train the model.
```Bash
python train.py
```
Run `test.py` to detect the following system behavior.
```Bash
python test.py
```
Run `anomly_time_windows.py` to process the detected data into time windows
```Bash
python anomly_time_windows.py
```
Run `eval.py` to evaluate the overall results
```Bash
python eval.py
```


