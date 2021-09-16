# ProteInfer

ProteInfer is an approach for predicting the functional properties of protein
sequences using deep neural networks.

## Usage instructions

### Install gcloud on your local machine if you don't have it installed
```
sudo apt install -y google-cloud-sdk
gcloud auth login
```

### Create GCP instance with a GPU
```
gcloud compute instances create proteinfer-gpu --machine-type n1-standard-8 --zone us-west1-b --accelerator type=nvidia-tesla-v100,count=1  --image-family ubuntu-2004-lts --image-project ubuntu-os-cloud --maintenance-policy TERMINATE --boot-disk-size 250
```

### ssh into the machine
```
# You may need to wait ~30 seconds for the machine to boot up first.
gcloud compute ssh proteinfer-gpu
```

### Install cuda dependencies for GPU support
```
sudo apt update
sudo add-apt-repository ppa:graphics-drivers -y
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo bash -c 'echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list'
sudo bash -c 'echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda_learn.list'

sudo apt update
sudo apt install -y cuda-10-0 libcudnn7
```

### Install local python virtual environment
```
sudo apt update
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt install -y python3-venv python3.7 python3-pip python3.7-venv 
mkdir ~/python_venv
cd ~/python_venv
python3.7 -m venv proteinfer
source ~/python_venv/proteinfer/bin/activate
cd ~
```

### Get our code from github and install python dependencies (e.g. numpy)
```
git clone https://github.com/google-research/proteinfer
cd ~/proteinfer
pip3 install -r requirements.txt
```

### Run our code on test sequences
```
cd ~/proteinfer
python3 install_models.py --install_ensemble
python3 proteinfer.py -i testdata/test_hemoglobin.fasta -o ~/hemoglobin_predictions.tsv --num_ensemble_elements=5

# View your predictions.
cat ~/hemoglobin_predictions.tsv
```

### Copy your sequences as a fasta file to the GCP instance
```
# exit the ssh session by typing ctrl D
gcloud compute scp <YOUR_FASTA_FILE_HERE> proteinfer-gpu:~/
# Then ssh back in again
gcloud compute ssh proteinfer-gpu
# Then run your inference
python3 proteinfer.py -i ~/<YOUR_FASTA_FILE_HERE> -o ~/predictions.tsv --num_ensemble_elements=5
```

### Delete the instance when you're done to save money
```
gcloud compute instances delete 'proteinfer-gpu'
```


## Running unit tests
```
bash -c 'for f in *_test.py; do python3 $f || exit 1; done'
```

## Status

This repository is still a work in progress. Please check back for more
documentation and a manuscript very soon. We are not able to accept pull 
requests or contributions at this time.
