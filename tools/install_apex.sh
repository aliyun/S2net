apt install libpython3.8-dev
git clone https://github.com/NVIDIA/apex

cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd .. && rm -rf apex
python3 -m pip install yacs
