# acompile -n 4 --time=02:00:00
# module load anaconda
# sinteractive --partition=aa100 --ntasks=20 --time=01:00:00 --gres=gpu:1 # for 1 gpu
# nvidea-smi

# partitions: a100, single a100 (takes less time to access), ami100
# scontrol show partitiion ami100 (less common, underutilized, not CUDA, amd equivalent
# nvidia-smi => Memory-Usage, CUDA Version, GPU_Util
# top -u $USER => to moniter
# ssh c3gpu-c2-u1 => to ssh into gpu node directly
# os.environ['CUDA_VISIBLE_DEVICES'] = 0 => use all visible devices
# sinteractive --partition=ami100 --ntasks=20 --time=01:00:00 --gres=gpu:1

conda create -n rl_wf_env python=3.8
conda activate rl_wf_env
#conda install python==3.8
pip install --no-cache-dir floris==2.4
# cd floridyn
python ./floridyn/setup.py develop
# cd ..
pip install --no-cache-dir torch
#conda install grpcio=1.43.0
pip install --no-cache-dir gymnasium
pip install --no-cache-dir scikit-learn
pip install --no-cache-dir dm_tree
#pip install --no-cache-dir ray[rllib]
#pip install --no-cache-dir ray[tune]
pip install --no-cache-dir gputil
pip install --no-cache-dir pettingzoo
pip install --no-cached-dir lightning