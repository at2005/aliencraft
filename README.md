We introduce AlienCraft, a procedural world generator that samples a new latent physics and chemistry for every world. The setup is simple:

- 2D Grid Worlds
- An abstract type system over materials: each "type" occupies a cell in the grid world. A type has a 3-dimensional property vector that maps to its base colour. Types can be crafted using tech-tree rules the agent must discover. Types are distributed in space using Perlin noise.
- Each world has N scalar fields partially observable to the agent:
  - Materials couple to fields via an affinity vector. The coupling is symmetric in that matter sources fields and fields influence matter.
  - Local field values tint the types' colour, which is what makes fields partially observable
  - To prevent the fields from exploding, we add a field damping factor
  - To prevent the fields from freezing into static states, we add a sinusoidal term that oscillates how matter influences field values.
- A craft is a commuting operation between two types. Type A + Type B → Type C, where C has properties inherited from A and B using the equation: P_c = tanh(T(p_a, p_b)) where T is a bilinear map sampled per-universe. The child's field affinities are similarly inherited.
  - Crafts succeed if the local affinity-weighted field magnitude exceeds the activation energy, given by the absolute sum of the two masses relative to the ambient field level.

We reward the agent for finding new types, similar to Crafter. The goal is to incentivise the agent to ascend its tech tree.

During generation we filter out "too boring" and "too chaotic" universes by rolling out universe dynamics and passing them through a gzip filter band.
