# Hard exploration in RL
This is an implementation for a project at the MVA master at ENS Paris-Saclay in the course Computer vision and Object recognition, and it consists of proposing a solution for the hard exploration problem in RL in the case of sparse rewards setting. The implementation is an extention of the Robin Strudel [main repository](https://github.com/rstrudel/nmprepr).
**The implementation can be found in the branch recvis20**.
Please find the instruction to install packages in the main branch (be aware to use the `environment.yml` of this branch), as well as basic instructions to run a standard training.

### Curriculum Learning
To launch the training with one of the supported options of curriculum learning, you can laucnh the `nmp.train` file and specifying the argument `-option` to one of the following:

1. cur-v0: increase the number of obstacles during training
2. cur-v1: increase the grid size
3. cur-v2: increase the distance to goal

For example you can use the following command to launch the `cur-v1` option:
```
python -m nmp.train -env-name Maze-grid-v3 -start-grid-size 2 -exp-dir maze_grid_3_cur_v1 --replay-buffer-size 75000 --horizon 75 --seed 0 --epochs 1700 -cur-range 250 -max-grid-size 4 -range-log 1 -option cur-v1
```
### Learning from demonstrations
To launch a training from demonstration you can use the following command:
```
python -m nmp.train_demo -env-name Maze-grid-v3 -exp-dir maze_grid_3_bc --replay-buffer-size 75000 --horizon 75 --seed 0 --epochs 2000 -demo-path nmp/data/dataset_maze_5_3000_perfect.pkl --gamma-bc 1e-3 -batch-size-demo 64 -bc-dist False -use-filter False #-warm-up 30
```
The main arguments are:
-`gamma-bc`: the parameter multiplied by the supervised loss (BC Loss).
-`batch-size-demo`: demonstration batch size.
-`bc-dist`: to use a Soft version of the BC Loss.
-`use-filter`: to use a filtered BC Loss (in case of imperfect demonstration).

To generate a dataset from a pretrained policy, go to branch:`recvis_generate_data` and use the file `mmp/run.py` to generate it.

To generate a perfect dataset using the A-start algorithm, switch to branch:`recvis_generate_data` and use the file `Generate_data.py` to generate it.

There are other options (e.g. using pretrained models, using noise networks...) that can be used by specifying the corresponding arguments in the file `nmp/train.py` or `nmp/train_demo.py`.
