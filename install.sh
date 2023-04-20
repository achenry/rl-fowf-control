# acompile -n 4 --time=02:00:00
# module load anaconda
# sinteractive --partition=aa100 --ntasks=20 --time=01:00:00 --gres=gpu:1 # for 1 gpu
# nvidea-smi

conda install python==3.8
pip install --no-cache-dir floris==2.4
# cd floridyn
python ./floridyn/setup.py develop
# cd ..
pip install --no-cache-dir torch
conda install grpcio=1.43.0
pip install --no-cache-dir gymnasium
pip install --no-cache-dir scikit-learn
pip install --no-cache-dir dm_tree
pip install --no-cache-dir ray[rllib]
pip install --no-cache-dir ray[tune]
pip install --no-cache-dir gputil
pip install --no-cache-dir pettingzoo