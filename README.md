A meta-learning environment/benchmark based on the game of Minecraft/Crafter.

Instead of fixed materials, we initialise an abstract type system that populates procedurally generated 2D grid-worlds. Some of these types have affinities to "fields" that affect their movement and interactions. In turn, types can generate fields. Fields are continuous scalar fields, while types live on a discrete grid.

Types can be combined to create new types with different properties/affinities through a crafting system. The agent is given a "craft" action that allows it to combine two types beside it into a new type.

Unlike Crafter, however, the agent does not have any survival constraint. It is effectively unbounded and progress is measured by how far down the tech tree it can get.

The meta-learning task is to understand the tech tree of a novel world and progress down it. We can increase the difficulty of the meta-learning task by increasing the degrees of freedom in the universe ruleset generator.

While this is less ideal than a "true" alien universe, it gives us structure and meaningful worlds, versus pure chaos/randomness that plagued the Lagrangian/CA setups.