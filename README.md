# Advanced AI Search Algoritms with Pacman

Explore various search algorithms and heuristics to help Pacman navigate through mazes, find food, and solve challenging problems. This README provides an overview of the project, instructions for running the code, and details on each question you need to answer. 

## Getting Started

To get started with this project, follow these steps:

1. **Download the project code** (`search.zip`).
2. **Unzip the downloaded file**.
3. **Change your terminal directory** to the unzipped project folder.

## Playing Pacman

You can play Pacman by running the following command in your terminal:

```bash
python pacman.py
```

Pacman lives in a maze filled with dots, and your goal is to help Pacman navigate through the maze efficiently to collect all the dots. You can control Pacman using arrow keys.

## Project Questions

### Question 1: Finding a Fixed Food Dot using Depth First Search

In this question, you will implement the Depth-First Search (DFS) algorithm in the `depthFirstSearch` function in `search.py`. This algorithm will help Pacman find a path through the maze. You should test your implementation using the following command:

```bash
python pacman.py -l tinyMaze -p SearchAgent -a fn=tinyMazeSearch
```

### Question 2: Breadth First Search

In this question, you will implement the Breadth-First Search (BFS) algorithm in the `breadthFirstSearch` function in `search.py`. Similar to Question 1, you will use BFS to find a path through the maze. Test your implementation with commands like:

```bash
python pacman.py -l mediumMaze -p SearchAgent -a fn=bfs
```

### Question 3: Varying the Cost Function

Implement the Uniform-Cost Search (UCS) algorithm in the `uniformCostSearch` function in `search.py`. This algorithm allows you to vary the cost function to encourage Pacman to find different paths. Test your implementation with commands like:

```bash
python pacman.py -l mediumMaze -p SearchAgent -a fn=ucs
```

### Question 4: A* Search

Implement the A* graph search in the `aStarSearch` function in `search.py`. Use the Manhattan distance heuristic (`manhattanHeuristic`) to find an optimal path. Test your implementation with commands like:

```bash
python pacman.py -l bigMaze -z .5 -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic
```

### Question 5: Finding All the Corners

In this question, you will help Pacman find the shortest path through the maze that touches all four corners. Implement the `CornersProblem` search problem in `searchAgents.py` and test it with commands like:

```bash
python pacman.py -l mediumCorners -p SearchAgent -a fn=bfs,prob=CornersProblem
```

### Question 6: Corners Problem: Heuristic

Implement a non-trivial, consistent heuristic for the `CornersProblem` in `cornersHeuristic`. Test your implementation with commands like:

```bash
python pacman.py -l mediumCorners -p AStarCornersAgent -z 0.5
```

### Question 7: Eating All The Dots

In this final question, you will help Pacman eat all the food dots in as few steps as possible. Implement the `foodHeuristic` in `searchAgents.py` and test it with commands like:

```bash
python pacman.py -l trickySearch -p AStarFoodSearchAgent
```

## Additional Notes

- Make sure to read the project instructions and comments in the code for more details on each question.
- Use the provided data structures in `util.py` for your implementations.
- Keep in mind the concepts of admissibility and consistency when designing heuristics.
- You can refer to the lecture slides for pseudocode and algorithm details.

Feel free to explore and experiment with different algorithms and heuristics to improve Pacman's performance. Good luck and have fun playing with Pacman!