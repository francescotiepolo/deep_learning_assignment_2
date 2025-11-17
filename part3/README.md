# Assignment 2 part 3: Graph Neural Networks
### Tasks
- Implement the class methods in the file [graph_cnn.py](graph_cnn.py).
  - MessageGraphConvolution Q3.2b
  - MatrixGraphConvolution Q3.3b
  - GraphAttention Q3.4d
- Do not put the activation function inside the layers, the activation functions are in the [model.py](model.py).
- Report the train and validation accuracy graphs
### Before running the program
There are new dependencies in this part of the assignment.
You can update the environment using:
```bash
conda env update -f dl2025_gpu.yml
```
These new depencies are introduced by the NeighborLoader. This data loader provides labels only for certain nodes (e.g. train set or minibatch), but the inputs for all nodes connected to those nodes. This is a necessary precaution for Graph networks, where splitting the dataset is not trivial. 
### Running the program
You can run all the necessary training with these commands: 
```bash
python train.py --model 'gcn'
python train.py --model 'matrix-gcn'
python train.py --model 'gat'
```
You can access the plots using Tensorboard:
```bash
tensorboard --logdir=logs
```

Feel free to use the provided unit tests:
```bash
python public_unittests.py
```
