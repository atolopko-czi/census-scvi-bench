sudo apt update
sudo apt install python3.10-venv emacs
python3.10 -m venv venv
. ~/venv/bin/activate
git clone https://gist.github.com/atolopko-czi/c2bb6f62173fea060764256030b59aeb
ln -s c2bb6f62173fea060764256030b59aeb/ census_scvi
cd census_scvi
pip install -r requirements.txt
python census_scvi.py