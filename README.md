

# A-LIX on DMC

PyTorch implementation of **Adaptive Local Signal Mixing** from **Stabilizing Off-Policy Deep Reinforcement Learning from Pixels**. This repository can be used to reproduce DeepMind Control experiments. 

For further details see our *ICML 2022* paper:


## Instructions

Install [MuJoCo](http://www.mujoco.org/)

Install dependencies with [conda](https://www.anaconda.com/):
```sh
conda env create -f conda_env.yml
conda activate drqv2
```

## Train an agent/collect results

Use Hydra configuration files (provided in the `cfgs` folder), specifying `algo` and `env`, 
representing the algorithm and environment configurations.

E.g.:
```sh
python train.py algo=ALIX task=quadruped_walk
```

You can monitor with tensorboard by running:
```sh
tensorboard --logdir exp_local
```

## Extend/contact

The main classes/functions used for A-LIX are located in the `analysis_*` files.

For any queries/questions, feel free to raise an issue and/or get in contact with [Edoardo Cetin](edoardo.cetin@kcl.ac.uk) or [Philip J. Ball](ball@robots.ox.ac.uk).

To cite our work, use:

```sh
@inproceedings{cetin2022stabilizing,
  title={Stabilizing Off-Policy Deep Reinforcement Learning from Pixels},
  author={Cetin, Edoardo and Ball, Philip J and Roberts, Stephen and Celiktutan, Oya},
  booktitle={International Conference on Machine Learning},
  pages={2784--2810},
  year={2022},
  organization={PMLR}
}
```

## Acknowledgements

We would like to thank Denis Yarats for open-sourcing the [DrQv2 codebase](https://github.com/facebookresearch/drqv2). Our implementation builds on top of their repository.
