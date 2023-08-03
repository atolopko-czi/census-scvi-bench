sudo apt update
sudo apt install -u software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.10-venv
python3.10 -m venv venv
. ~/venv/bin/activate
git clone https://github.com/atolopko-czi/census-scvi-bench.git
cd census-scvi-bench
pip install -r requirements.txt
python census_scvi.py
