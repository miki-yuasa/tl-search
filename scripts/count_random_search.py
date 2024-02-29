import numpy as np

node_size: int = 3 * 4 + 2
num_episodes: int = 10000

num_possible_nodes: int = 2**node_size
print(f"Number of possible nodes: {num_possible_nodes}")

episode_lengths: list[int] = []

for i in range(num_episodes):
    episode_length: int = 0

    current_node: tuple[int, ...] = tuple(np.random.randint(2, size=node_size).tolist())
    target_node: tuple[int, ...] = tuple(np.random.randint(2, size=node_size).tolist())

    # print(f"Current node: {current_node}")
    # print(f"Target node: {target_node}")

    searched_nodes: list[tuple[int, ...]] = [current_node]

    while True:
        episode_length += 1

        # Flip one value of the current node to create a new node
        flip_idx: int = np.random.randint(node_size)
        new_node: tuple[int, ...] = (
            current_node[:flip_idx]
            + (1 - current_node[flip_idx],)
            + current_node[flip_idx + 1 :]
        )

        if new_node == target_node or len(searched_nodes) == num_possible_nodes:

            break

        searched_nodes.append(new_node)
        current_node = new_node

    episode_lengths.append(episode_length)

print(f"Average episode length: {np.mean(episode_lengths)}")
print(f"Searched percentage: {np.mean(episode_lengths) / num_possible_nodes * 100}%")
