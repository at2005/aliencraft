> Research Preview (v0.1)

# AlienCraft: A Meta-Reinforcement-Learning Environment Generator for Open-Ended Explorers

## Introduction

A long-held goal of reinforcement learning has been to create agents that are able to navigate new worlds at test time without being explicitly trained on them. How do we evaluate progress towards this north star, this meta reinforcement-learner? DeepMind’s Alchemy (Wang et al., 2021) is an environment in which an agent must learn an abstract sampled “chemistry” at test time. However, Alchemy samples from a fixed parameterized family, which limits environment diversity. We’re releasing a research preview of AlienCraft, a procedural open-ended world generator that samples a new latent physics and chemistry for every world. We expect AlienCraft to be changed as people train models on it and we identify points of improvement.

## Motivation

If we are to believe in the “era of experience”, a desired ability is for an agent to build a model of a new world at test-time given some exploration budget. Any process that instills these capabilities during training should incentivise the emergence of meta-abilities like "running experiments" and "doing science", as opposed to memorising the dynamics of any particular world.

How do we train meta reinforcement learners? One way to do so is to train them directly in the real world across a large variety of tasks. However, this is impractical: performing rollouts in the real world is expensive, and we lack a high-fidelity simulator of human civilization. Furthermore, even though the real world has high complexity, for superhuman meta-learners we want to train them in domains that would be extremely difficult for humans, such that human tasks don’t bound the reasoning abilities of our agents.

Another way to train meta reinforcement learners is to build a procedural generator of tasks, such that each pair of tasks is maximally diverse, yet they have shared meta-structure. Whether or not the agent develops general reasoning abilities is constrained by this meta-structure. Our perspective for how to build such a generator is that there exists a continuum: on one extreme are environments that are pretty much the same, e.g. Crafter, and on the other are environments that are completely alien from each other. Given bounded computational resources we can’t “bitter-lesson-search” the space of all possible environments, but we can make progress on the continuum: environments that differ from one another but with a somewhat constrained rule-space.

## AlienCraft

Motivated by these design principles we introduce AlienCraft, an open-ended 2D grid-world environment similar to Crafter but where the physics and chemistry is sampled per-episode.

### Materials

Each cell is either empty or contains a material, initially distributed with Perlin noise. A material is defined by a three-dimensional property vector that maps to its base colour. Materials can be crafted using tech-tree rules the agent must discover. There are a finite number of possible materials in any given world.

### Fields

Each world possesses three scalar fields. These are partially observable to the agent, with local field values tinting materials’ base colors.

Materials couple to fields via an affinity vector. The coupling is symmetric in that matter sources fields and fields influence matter. Emissions from matter are deposited in its neighbourhood using a sampled 7x7 radially-symmetric kernel. The influence of matter on fields is determined by a 3x3 sampled kernel. This parameterization gives us complex fields with the inductive bias of locality.

### Crafting

A craft is a commuting operation between two materials. Material 1 + Material 2 → Material 3, where C has properties inherited from A and B using the equation: $p3_i = f(p1^T M_i p2)$ where $M_i$ is a bilinear map sampled per-universe, $i$ is an index running along the dimensions of the property vector, and $f$ is a sampled, bounded nonlinearity. The child’s field affinities are similarly inherited.

How does an agent craft? It places the two materials beside each other, and selects the craft action. A craft succeeds iff:

- $g(p1^T F p2) > threshold$ (set so that 25-40% of all pairs are craftable). $F$ is symmetric by construction.
- $h(z^T G z) > 0$, where $z = [EMA-normalized local fields, p1_affinity + p2_affinity, p1_properties + p2_properties]$

Both $h$ and $g$ are sampled as bounded nonlinearities, similar to $f$. If the resulting material properties are within epsilon of an existing material, we craft that material. Otherwise we treat it as a new material.

We reward the agent for crafting new materials, to incentivise climbing tech trees.

### Agent

The agent is given a 2D position in the grid, though it is not embodied. The agent takes as input an RGB frame of its local area centered on its position, and outputs a continuous action vector. We make action selection nontrivial by passing this continuous vector through a sampled invertible projection, using a Wolpertinger-like mechanism to map this to a discrete action. To select a material from the inventory, the agent outputs a 3-dimensional vector along which a nearest-neighbour lookup is performed with inventory material properties.

### What must the agent learn?

When placed in a new world, an agent must figure out:

- Its action mapping, ie the actuators
- Sampled nonlinear functions for crafting, property/affinity inheritance etc
- Sampled bilinear maps for crafting, property/affinity inheritance etc
- The cellular-automata-style kernels driving field evolution
- Property/affinity vectors for the initial, base materials

Since understanding these is crucial for a) climbing the tech tree, and b) for reducing world modelling loss, we expect model-based RL agents to develop meta-abilities such as building experimental apparatuses, carrying out experiments, etc.

We also develop an oracle that has full knowledge of the universe, infinite inventory, and the ability to teleport. While oracle success does not guarantee 
the environment to be solvable, failure does mean something is deeply broken. We did not find any universes produced that failed the oracle test.

## Complexity Filters

While we do have some constraints in our rule-space, we find that our “sample as much as we can” approach has some problems. Namely, the prevalence of dead or unstable universes. To mitigate this problem we add a series of filters:

- Gzip ratio of random trajectories within [0.1, 0.65], prevents frozen or chaotic worlds
- Linearity of fields: we fit a 3x3 kernel to predict the next field state. If the fit is highly accurate, the transition is linear (undesirable).
- Matter mobility: does matter move? We check if the fraction of occupied cells whose matter moves through the trajectory is within [0.05, 0.95].
- Sensitivity of fields to matter: start with two identical universes, but one of them has the influence of matter zeroed out. How much do they diverge? Check if the RMS divergence after T steps is > 0.05. This rules out universes where fields dominate and matter has no influence.
- Field noise-level: we measure the coefficient of determination to see how much of the field variation is explained by a simple “copy the previous frame” predictor. Values near 1 indicate frozen worlds, while values near -1 are white noise. We bound this statistic to be within [0.2, 0.995]
- Tech tree depth >= 3
- Fraction of gates open for any craftable pair along a T-step horizon must be >= 0.999
- In the worst case, at least 25% of gates are open somewhere
- Median gate open area must be less than 60% of the total grid. If gates are open everywhere this removes the location barrier to crafting.
- The median per-channel color difference between materials >= 30
- Number of possibly craftable materials within [10, 300]

## Using the Gym Wrapper

Gymnasium support is an optional extra:

```bash
pip install -e ".[gym]"
```

Importing `aliencraft` registers `AlienCraft-v0`:

```python
import gymnasium as gym
import aliencraft  # registers AlienCraft-v0

env = gym.make("AlienCraft-v0")
obs, info = env.reset(seed=0)  # seed controls world generation
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

- **Observation**: `(visual_field_size, visual_field_size, 3)` float32 RGB in [0, 1], centered on the agent (32x32 by default).
- **Action**: continuous `Box(-1, 1)` of size `2 + num_properties` (5 by default): 2 motion dimensions plus a property-space pointer used for the nearest-neighbour inventory lookup.
- **Reward**: +1 for each newly discovered material.
- **Episodes**: truncated at `max_episode_steps` (default 1000); there is no terminal state.
- `info` reports `discovered_types` and `agent_position`.

Useful `gym.make` kwargs:

- `max_episode_steps`, `device`, and any `AlienCraftWorld` kwarg (e.g. `visual_field_size=64` for CNN policies that need larger inputs).
- `complexity_band` — gzip-ratio band for the complexity filters (default `(0.1, 0.65)`); pass `None` to sample unfiltered worlds.
- `pool` — path to a `torch.save`d pool of pre-filtered laws; each reset draws from the pool instead of sampling and filtering a fresh world.
- `render_mode="rgb_array"` — `env.render()` returns a full-world uint8 frame.

See `baselines/ppo_sb3.py` for a complete example training a PPO agent with stable-baselines3.
