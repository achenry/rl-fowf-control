conda install python==3.8
pip install floris==2.4
cd floridyn
python setup.py develop
cd ..
pip install torch
conda install grpcio=1.43.0
pip install gymnasium
pip install scikit-learn
pip install dm_tree
pip install ray[rllib]
pip install ray[tune]
pip install gputil