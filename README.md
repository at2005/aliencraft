# AlienCraft: A Meta-Reinforcement-Learning Environment Generator for Open-Ended Explorers

> **Research Preview (v0.1)**

## Introduction

A long-held goal of reinforcement learning has been to create agents that are able to navigate new worlds at test time without being explicitly trained on them. How do we evaluate progress towards this north star, this meta-reinforcement learner?

DeepMind’s Alchemy (Wang et al., 2021) is an environment in which an agent must learn an abstract sampled “chemistry” at test time. However, Alchemy samples from a fixed parameterized family, which limits environment diversity.

We’re releasing a research preview of AlienCraft, a procedural open-ended world generator that samples a new latent physics and chemistry for every world. We expect AlienCraft to change as people train models on it and we identify points of improvement.

## Motivation

If we are to believe in the “era of experience,” a desired ability is for an agent to build a model of a new world at test time given some exploration budget. Any process that instills these capabilities during training should incentivise the emergence of meta-abilities like “running experiments” and “doing science,” as opposed to memorising the dynamics of any particular world.

How do we train meta-reinforcement learners? One way to do so is to train them directly in the real world across a large variety of tasks. However, this is impractical: performing rollouts in the real world is expensive, and we lack a high-fidelity simulator of human civilization. Furthermore, even though the real world has high complexity, for superhuman meta-learners we want to train them in domains that would be extremely difficult for humans, such that human tasks don’t bound the reasoning abilities of our agents.

Another way to train meta-reinforcement learners is to build a procedural generator of tasks, such that each pair of tasks is maximally diverse, yet they have shared meta-structure. Whether or not the agent develops general reasoning abilities is constrained by this meta-structure.

Our perspective for how to build such a generator is that there exists a continuum: on one extreme are environments that are pretty much the same, such as Crafter, and on the other are environments that are completely alien from each other. Given bounded computational resources, we can’t “bitter-lesson-search” the space of all possible environments, but we can make progress along the continuum: environments that differ from one another but have a somewhat constrained rule space.

## AlienCraft

Motivated by these design principles, we introduce AlienCraft, an open-ended 2D grid-world environment similar to Crafter but where the physics and chemistry are sampled per episode.

### Materials

Each cell is either empty or contains a material, initially distributed with Perlin noise. A material is defined by a three-dimensional property vector that maps to its base colour. Materials can be crafted using tech-tree rules the agent must discover. There are a finite number of possible materials in any given world.

### Fields

Each world possesses three scalar fields. These are partially observable to the agent, with local field values tinting materials’ base colours.

Materials couple to fields via an affinity vector. The coupling is symmetric in that matter sources fields and fields influence matter. Emissions from matter are deposited in its neighbourhood using a sampled $7 \times 7$ radially symmetric kernel. The influence of matter on fields is determined by a sampled $3 \times 3$ kernel. This parameterization gives us complex fields with the inductive bias of locality.

### Crafting

A craft is a commuting operation between two materials:

$$
\text{Material 1} + \text{Material 2} \rightarrow \text{Material 3}.
$$

The child material has properties inherited from its parents using:

$$
p_{3,i} = f\!\left(p_1^\top M_i p_2\right),
$$

where $M_i$ is a bilinear map sampled per universe, $i$ indexes the dimensions of the property vector, and $f$ is a sampled, bounded nonlinearity. The child’s field affinities are inherited similarly.

To craft, the agent places two materials beside each other and selects the craft action. A craft succeeds if and only if both of the following conditions hold:

1. The material pair is craftable:

   $$
   g\!\left(p_1^\top Fp_2\right) > \text{threshold},
   $$

   where the threshold is set so that 25–40% of all pairs are craftable and $F$ is symmetric by construction.

2. The local field gate is open:

   $$
   h\!\left(z^\top Gz\right) > 0,
   $$

   where

   $$
   z = [\text{EMA-normalized local fields},\ p_{1,\mathrm{affinity}} + p_{2,\mathrm{affinity}},\ p_{1,\mathrm{properties}} + p_{2,\mathrm{properties}}].
   $$

Both $h$ and $g$ are sampled bounded nonlinearities, similar to $f$. If the resulting material properties are within $\varepsilon$ of an existing material, we craft that material. Otherwise, we treat it as a new material.

We reward the agent for crafting new materials, incentivising it to climb the tech tree.

### Agent

The agent is given a 2D position in the grid, though it is not embodied. It takes as input an RGB frame of its local area centred on its position and outputs a continuous action vector.

We make action selection nontrivial by passing this continuous vector through a sampled invertible projection, using a Wolpertinger-like mechanism to map it to a discrete action. To select a material from the inventory, the agent outputs a three-dimensional vector along which a nearest-neighbour lookup is performed using inventory material properties.

### What Must the Agent Learn?

When placed in a new world, an agent must determine:

- Its action mapping—that is, its actuators.
- The sampled nonlinear functions used for crafting and property and affinity inheritance.
- The sampled bilinear maps used for crafting and property and affinity inheritance.
- The cellular-automata-style kernels driving field evolution.
- The property and affinity vectors of the initial base materials.

Understanding these is crucial both for climbing the tech tree and for reducing world-modelling loss. We therefore expect model-based RL agents to develop meta-abilities such as building experimental apparatuses and carrying out experiments.

### Oracle

We also develop a perfect oracle that has full knowledge of the universe, infinite inventory, and the ability to teleport. We find that this oracle solves more than 95% of all universes generated.

## Complexity Filters

While we impose some constraints on our rule space, our “sample as much as we can” approach produces a prevalence of dead or unstable universes. To mitigate this problem, we apply a series of filters:

- **Trajectory compressibility:** The gzip ratio of random trajectories must lie within $[0.1, 0.65]$, preventing frozen or chaotic worlds.

- **Field linearity:** We fit a $3 \times 3$ kernel to predict the next field state. If the fit is highly accurate, the transition is linear, which is undesirable.

- **Matter mobility:** The fraction of occupied cells whose matter moves during the trajectory must lie within $[0.05, 0.95]$.

- **Sensitivity of fields to matter:** We start with two identical universes but zero out the influence of matter in one. The RMS divergence after $T$ steps must be greater than $0.05$. This rules out universes where fields dominate and matter has no influence.

- **Field noise level:** We measure the coefficient of determination to determine how much field variation is explained by a simple “copy the previous frame” predictor. Values near $1$ indicate frozen worlds, while values near $-1$ indicate white noise. We require this statistic to lie within $[0.2, 0.995]$.

- **Tech-tree depth:** The tech-tree depth must be at least $3$.

- **Gate availability over time:** The fraction of gates open for any craftable pair along a $T$-step horizon must be at least $0.999$.

- **Worst-case gate availability:** At least 25% of gates must be open somewhere, even in the worst case.

- **Gate locality:** The median gate-open area must be less than 60% of the total grid. If gates are open everywhere, the location barrier to crafting disappears.

- **Material distinguishability:** The median per-channel colour difference between materials must be at least $30$.

- **Craftable material count:** The number of possibly craftable materials must lie within $[10, 300]$.
