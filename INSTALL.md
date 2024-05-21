## Installation

### Requirements:
- PyTorch 
- torchvision 
- opencv-python
- python 3.7
- yacs
- matplotlib
- pycocotools
- cityscapesscripts
- tqdm
- scipy
- mmcv
- apex
- GCC
- OpenCV
- CUDA


### Option : Step-by-step installation

```bash
conda create -n lstfe python=3.7.16

source activate lstfe

pip install yacs opencv-python pycocotools cityscapesscripts scipy

pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 --extra-index-url https://download.pytorch.org/whl/cu113

