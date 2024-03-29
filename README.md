# Advanced AI Algorithms with Pacman

## Getting Started

To get started with this project, follow these steps:

1. **Download the project code** (`ai-main.zip`).
2. **Unzip the downloaded file**.
3. **Change your terminal directory** to the unzipped project folder.

## 1. Advanced Search 
```
cd ai-main\1_advanced_search
```

### Depth First Search to Find a Path to a Fixed Object

`depthFirstSearch` function in `search.py` finds a path through the maze. You can run this implementation using the following command:

```bash
python pacman.py -l tinyMaze -p SearchAgent -a fn=tinyMazeSearch
```

https://github.com/narissatsuboi/ai/assets/79029751/6008466e-8f1f-4f59-b7bf-badf871e4b8e


### Breadth First Search to Find a Path to a Fixed Object
`breadthFirstSearch` function in `search.py` uses BFS to find a path through the maze. You can run this implementation using the following command:

```bash
python pacman.py -l mediumMaze -p SearchAgent -a fn=bfs
```

### Uniform-Cost Search to Find a Path to a Fixed Object

`uniformCostSearch` function in `search.py`allows you to vary the cost function to encourage Pacman to find different paths. You can run this implementation using the following command:

```bash
python pacman.py -l mediumMaze -p SearchAgent -a fn=ucs
```

### A* Search

`aStarSearch` function in `search.py` uses the Manhattan distance heuristic (`manhattanHeuristic`) to find an optimal path. You can run this implementation using the following command:

```bash
python pacman.py -l bigMaze -z .5 -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic
```

### A* Search to Find All the Corners

`CornersProblem` search problem in `searchAgents.py` uses A* search to find the shortest path through the maze that touches all four corners. You can run this implementation using the following command:

```bash
python pacman.py -l mediumCorners -p SearchAgent -a fn=bfs,prob=CornersProblem
```

### Corners Problem: Heuristic

`CornersProblem` search problem in `searchAgents.py` uses a consistent heuristic. You can run this implementation using the following command:

```bash
python pacman.py -l mediumCorners -p AStarCornersAgent -z 0.5
```

### Eating All The Dots : Heuristic

`foodHeuristic` in `searchAgents.py` helps Pacman eat all the food dots in as few steps as possible. You can run this implementation using the following command:

```bash
python pacman.py -l trickySearch -p AStarFoodSearchAgent
```

## 2. Game Trees
```
cd ai-main\1_game_trees
```
### Reflex Agent 
`ReflexAgent` in `multiagents.py` considers food and ghost locations to maximize the food it reaches without running into an active ghost. 
```bash
python pacman.py --frameTime 0 -p ReflexAgent -k 2
```

### Minimax Adversarial Agent
`MinimaxAgent` in `multiagents.py` is a generalized algorithm to play against any number of ghosts when they always play optimally. 
```bash
python pacman.py -p MinimaxAgent -l minimaxClassic -a depth=4
```

## Alpha-Beta Pruning
`AlphaBetaAgent` in `multiagents.py` uses alpha-beta pruning to efficiently traverse the minimax tree when ghosts always play optimally. 
```bash
python pacman.py -p AlphaBetaAgent -a depth=3 -l smallClassic
```

## Expectimax 
`ExpectimaxAgent` in `multiagents.py` uses Q values to play well against ghosts who don't always play optimally. 
