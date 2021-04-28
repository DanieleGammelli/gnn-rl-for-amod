# Graph Neural Network Reinforcement Learning for AMoD Systems
Official implementation of [Graph Neural Network Reinforcement Learning for Autonomous Mobility-on-Demand Systems](https://arxiv.org/abs/2104.11434)

<img align="center" src="images/gnn-for-amod.png" width="700"/></td> <br/>

## Prerequisites

You will need to have a working IBM CPLEX installation. If you are a student or academic, IBM is releasing CPLEX Optimization Studio for free. You can find more info [here](https://community.ibm.com/community/user/datascience/blogs/xavier-nodet1/2020/07/09/cplex-free-for-students)

To install all required dependencies, run
```
pip install -r requirements.txt
```

## Contents

* `src/algos/a2c_gnn.py`: PyTorch implementation of A2C-GNN.
* `src/algos/reb_flow_solver.py`: thin wrapper around CPLEX formulation of the Rebalancing problem (Section III-A in the paper).
* `src/envs/amod_env.py`: AMoD simulator.
* `src/cplex_mod/`: CPLEX formulation of Rebalancing and Matching problems.
* `src/misc/`: helper functions.
* `data/`: json files for NYC experiments.
* `saved_files/`: directory for saving results, logging, etc.

## Examples

To train an agent, `main.py` accepts the following arguments:
```bash
cplex arguments:
    --cplexpath     defines directory of the CPLEX installation
    
model arguments:
    --test          activates agent evaluation mode (default: False)
    --max_episodes  number of episodes to train agent (default: 16k)
    --max_steps     number of steps per episode (default: T=60)
    --no-cuda       disables CUDA training (default: True, i.e. run on CPU)
    --directory     defines directory where to log files (default: saved_files)
    
simulator arguments: (unless necessary, we recommend using the provided ones)
    --seed          random seed (default: 10)
    --demand_ratio  (default: 0.5)
    --json_hr       (default: 7)
    --json_tsetp    (default: 3)
    --no-beta       (default: 0.5)
```

**Important**: Take care of specifying the correct path for your local CPLEX installation. Typical default paths based on different operating systems could be the following
```bash
Windows: "C:/Program Files/ibm/ILOG/CPLEX_Studio128/opl/bin/x64_win64/"
OSX: "/Applications/CPLEX_Studio128/opl/bin/x86-64_osx/"
Linux: "/opt/ibm/ILOG/CPLEX_Studio128/opl/bin/x86-64_linux/"
```
### Training and simulating an agent

1. To train an agent (with the default parameters) run the following:
```
python main.py
```

2. To evaluate a pretrained agent run the following:
```
python main.py --test=True
```

## Credits
This work was conducted as a joint effort with [Kaidi Yang*](https://sites.google.com/site/kdyang1990/), [James Harrison*](https://stanford.edu/~jh2/), [Filipe Rodrigues'](http://fprodrigues.com/), [Francisco C. Pereira'](http://camara.scripts.mit.edu/home/) and [Marco Pavone*](https://web.stanford.edu/~pavone/), at Technical University of Denmark' and Stanford University*. 

## Reference
```
@inproceedings{GammelliYangEtAl2021,
  author = {Gammelli, D. and Yang, K. and Harrison, J. and Rodrigues, F. and Pereira, F. C. and Pavone, M.},
  title = {Graph Neural Network Reinforcement Learning for Autonomous Mobility-on-Demand Systems},
  year = {2021},
  note = {Submitted},
}
```

----------
In case of any questions, bugs, suggestions or improvements, please feel free to contact me at daga@dtu.dk.
