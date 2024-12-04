## SNSNet
Part of the official repo for paper "JBF: An Enhanced Representation of Skeleton for Video-based Human Action Recognition"

### Installation
1. Create environment:
    ```
    conda create -n sns python=3.9 -y
    conda activate sns
    ```
2. Install PyTorch (Change verions according to your CUDA version):
    ```
    conda install pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.8 -c pytorch -c nvidia
    ```
3. Install OpenMM:
    ```
    pip install -U openmim
    mim install mmengine "mmcv==2.1.0" "mmdet==3.2.0"
    ```
4. Install additional dependencies:
    ```
    pip install -r requirements.txt
    pip install -v -e .
    ```
5. (Optional, bug exists with certain CUDA versions) Install alternate correlation implementation from [RAFT]('https://github.com/princeton-vl/RAFT/tree/master'):
    ```
    cd mmpose/models/flownets/alt_cuda_corr
    python setup.py install
    cd ../../../..
    ```