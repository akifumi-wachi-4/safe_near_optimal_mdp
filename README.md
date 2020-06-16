# Safe Near-Optimal MDP (SNO-MDP)

This is the source-code for implementing the algorithms in the paper "Safe Reinforcement Learning in Constrained Markov Decision Processes" which was presented in ICML-20.

## Installation

The easiest way is to use the Anaconda Python distribution. Then, run the following commands to install the necessary packages:

### GPy and pymdptoolbox

First of all, we used <a href="https://github.com/SheffieldML/GPy" target="_blank">GPy</a> to implement Gaussian Processes (GPs) and <a href="https://github.com/sawcordwell/pymdptoolbox" target="_blank">pymdptoobox</a> to calculate the optimal policy for a given Markov Decision Process (MDP).

```
pip install GPy
pip install pymdptoolbox
```

### SafeMDP

Our code also depends upon <a href="https://github.com/befelix/SafeMDP" target="_blank">SafeMDP</a>. This module can be installed by the following commands.

```bash
git clone https://github.com/befelix/SafeMDP.git
cd SafeMDP
python setup.py install
```


### Safety-Gym

Finally, we developed a new environment called GP-Safety-Gym. This enviornment is based on OpenAI Safety-Gym. For the installation, see <a href="https://github.com/openai/safety-gym" target="_blank">Safety-Gym</a> repository. Note that OpenAI Safety-Gym and our GP-Safety-Gym heavily depends on  <a href="https://github.com/openai/mujoco-py" target="_blank">mujoco_py</a>.



## GP-Safety-Gym

<img src="./GPSG.png" width="400">

To use our GP-Safety-Gym environment, first define an `Engine_GP` environment by

```
env = Engine_GP(config, reward_map=reward, safety_map=safety)
```

and render 1) the agent's position and 2) safety and reward functions by

```
env.discreate_move(pos)
```

We also provide a sample script for running GP-Safety-Gym in `./test/test_gp_safety_gym.py`. 
In this script, the agent will target for randomly specified positions.



## Synthetic Environment

For running the synthetic experiment in our paper, you first create a random environment and a start position by the following command:

```python
python simple_make_rand_settings.py
```

And, run the simulation for each method:
```bash
python simple_oracle.py    # Safe/reward known
python simple_proposed.py  # SNO-MDP
python simple_safemdp.py   # SafeMDP (Turchetta et al., 2016)
python simple_seo.py       # SafeExpOpt-MDP (Wachi et al., 2018)
```

For our proposed method (i.e.,`simple_proposed.py`) , you can specify whether `ES2`/`P-ES2` is leveraged, using `simple_args.py`.

```python
parser.add_argument('--es2-type', type=str, default='es2', 
                    choices=['es2', 'p_es2', 'none'],
                    help='whether or not ES2/P-ES2 is used')
```

## Citation

If you find this code useful in your research, please consider citing:
```bibtex
@inproceedings{wachi_sui_snomdp_icml2020,
  Author = {Akifumi Wachi and Yanan Sui},
  Title = {Safe Reinforcement Learning in Constrained Markov Decision Processes},
  Booktitle  = {International Conference on Machine Learning (ICML)},
  Year = {2020}
}
```

