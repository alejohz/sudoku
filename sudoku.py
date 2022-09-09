import numpy as np
from random import shuffle, randint, uniform
from typing import List, Optional


def _transform(problem_grid: List[List]) -> List[List]:
    """
    Transforms the problem_grid passed, testing the sudoku has three main comparisons.
    Assuming we have an untransformed grid, we have:
        1. All rows must have each number from 1 to 9 with no repetition
        2. All columns must have each number from 1 to 9 with no repetition
        3. All 3x3 squares must have each number from 1 to 9 with no repetition
    These last conditions are assuming we have a 9x9 grid with 81 numbers in total.

    To be able to easily check for rows, columns and squares, we transform the grid, to convert each 3x3 square to a 1x9 row.
    These way in the untransformed version we check rows and columns
    In the transformed version we check rows, and now we have checked every condition.
    Parameters:
    - problem_grid (list): Sudoku grid (Transformed or Untransformed).
    Returns (list).
    - transformed grid .
        If problem_grid is the original sudoku it converts it
        If the problem_grid is the already transformed sudoku, it unrolls it
    """  # noqa: E501
    file_lines = problem_grid
    problem_grid = [[] for _ in range(len(file_lines))]  # Empty grid
    sqrt_n = 3  # this parameter could change if one wants to generalize for sudokus of other dimensions # noqa: E501
    for j in range(len(file_lines)):
        line_values = [(int(value) if value != 0 else None)
                       for value in file_lines[j]]
        for i in range(len(line_values)):
            problem_grid[
                int(i / sqrt_n) +
                int(j / sqrt_n) * sqrt_n
                ].append(line_values[i])
    return problem_grid


class Sudoku:
    def __init__(self,
                 grid: List[List]):
        self.grid = grid
        self.grid_length = len(self.grid)
        self.problem_grid = self._deep_copy_grid(_transform(self.grid))

    def _empty_grid(self,
                    elem_generator: List[List] | None = None) -> List[List]:
        """
        Returns an empty Sudoku grid.
        Parameters:
        - elem_generator (function) (optional=None): Is is used to generate initial values of the grid's elements.
                If it's not given, all grid's elements will be "None".
        """  # noqa: E501
        return [
            [
                (None if elem_generator is None else elem_generator(i, j))
                for j in range(self.grid_length)
            ] for i in range(self.grid_length)
        ]

    def _deep_copy_grid(self, grid: List[List]
                        ) -> List[List]:
        """
        Returns a deep copy of the grid argument.
        The reason why there is an empty grid and the a deep copy of a grid is the way python works with lists.
        For example, sample function and normal assignation of none to a list could modify the original list where
        it was defined from.
        By deep copying an empty grid we make sure, the original grid is not modified.
        Parameters:
            - grid (list)
        """  # noqa: E501
        return self._empty_grid(lambda i, j: grid[i][j])

    def _fitness(self,
                 grid: List[List]
                 ) -> int:
        """
        Calculates the fitness function for a grid, the fitness integer is the number of repeated elements,
        either in a row, column or square.
        Parameters:
            - grid (list)
        Returns (int): The value of the fitness function for the input grid.
        """  # noqa: E501
        grid = np.array(grid)
        grid_2 = _transform(grid)

        # COUNT HOW MANY REPEATED NUMBERS
        row = np.sum(9 - np.array([len(np.unique(r)) for r in grid_2]))
        column = np.sum(9 - np.array([len(np.unique(c))
                                      for c in np.transpose(grid_2)]))
        square = np.sum(9 - np.array([len(np.unique(r)) for r in grid]))

        # Verification of given values
        actual = (sum(x is not None for x
                      in np.array(self.problem_grid).flatten())
                  - np.sum(np.array(self.problem_grid).flatten()
                           == np.array(grid).flatten()))
        fitness = row + column + square + actual
        return fitness

    def _generate_initial_population(self,
                                     pop_size: int,
                                     ) -> List[List]:
        """
        Generates an initial population of size "population_size".
        Returns (list): An array of candidate grids.
        """
        candidates = []
        for _ in range(pop_size):
            candidate = self._empty_grid()
            for i in range(self.grid_length):
                shuffled_sub_grid = [n for n in
                                     range(1, self.grid_length + 1)]
                shuffle(shuffled_sub_grid)
                for j in range(self.grid_length):
                    if self.problem_grid[i][j] is not None:
                        candidate[i][j] = self.problem_grid[i][j]
                        shuffled_sub_grid.remove(self.problem_grid[i][j])
                for j in range(self.grid_length):
                    if candidate[i][j] is None:
                        candidate[i][j] = shuffled_sub_grid.pop()

            candidates.append(candidate)

        return candidates

    def _selection(self,
                   candidates: List[List],
                   selection_rate: float,
                   ) -> List[List]:
        """
        Returns the best portion ("selection_rate") of candidates based on their fitness function values (lower ones).
        Parameters:
          - candidates (list)
        Returns:
          - ordered candidates (list)
          - fitness of the first candidate (int)
          - fitness of the last candidate (int)
        """  # noqa: E501
        index_fitness = []
        for i in range(len(candidates)):
            index_fitness.append(tuple([i, self._fitness(candidates[i])]))
        index_fitness.sort(key=lambda elem: elem[1])
        selected_part = index_fitness[0: int(len(index_fitness)
                                             * selection_rate)]
        indexes = [e[0] for e in selected_part]
        return ([candidates[i] for i in indexes],
                selected_part[0][1],
                selected_part[-1][1]
                )

    def solve(self,
              population_size: Optional[int] = 10000,
              selection_rate: Optional[float] = 0.8,
              max_generations_count: Optional[int] = 100,
              mutation_rate: Optional[float] = 0.4,
              random_proportion: Optional[float] = 0.5,
              verbose: Optional[bool] = False,
              cross_over: Optional[str] = "best candidate",
              ):
        self.population_size = population_size
        self.selection_rate = selection_rate
        self.max_generations_count = max_generations_count
        self.mutation_rate = mutation_rate
        self.random_proportion = random_proportion
        """
        Solves a Sudoku puzzle using a genetic algorithm.
        It receives a problem grid written in different lines (Not one line with 81 numbers) and in the empty spaces, a 0 should appear
        In the "reading sudoku" chapter there is an example of how the initial sudoku should be formatted.

        Parameters:
            - problem_grid (list): An 9x9 sudoku grid.
            - population_size (int): The initial population size.
            - selection_rate (int)
            - max_generations_count (int)
            - mutation_rate (int)
        Returns:
            - population[0] (list): Solved sudoku.
            - best_fitness (int): last best_fitness, when not solved in the max_generations it gets the last one.
            - fitness_history (list): For plotting progress of learning
        """  # noqa: E501

        population = self._generate_initial_population(population_size)
        best_fitness = 100  # initializing value with an improbable value
        fitness_history = []  # for plotting
        for i in range(max_generations_count):
            (population,
             best_fitness,
             worst_fitness) = self._selection(population, self.selection_rate)
            if best_fitness == 0:
                print(f"Generation {i} -Best candidate's fitness {best_fitness} -Worst candidate's fitness {worst_fitness}")  # noqa: E501
                break
            actual_pop_size = len(population)
            new_population_size = int((population_size - actual_pop_size)
                                      * (1 - random_proportion))
            random_population_size = int((population_size - actual_pop_size)
                                         * random_proportion)
            new_population = []
            # Cross-over
            if cross_over == "best candidate":
                for _ in range(new_population_size):
                    # Selecting two random candidates
                    w1 = randint(0, actual_pop_size - 1)
                    w2 = randint(0, actual_pop_size - 1)
                    c1 = population[w1].copy()  # Copy to not modify original list  # noqa: E501
                    c2 = population[w2].copy()

                    # Random cross-over
                    cross_point = randint(0, 8)
                    temp = c2[cross_point]
                    c1[cross_point] = c2[cross_point]
                    c2[cross_point] = temp

                    if self._fitness(c1) < self._fitness(c2):
                        new_population.append(c1)
                    else:
                        new_population.append(c2)
            elif cross_over == "random candidate":
                for _ in range(new_population_size):
                    # Selecting two random candidates
                    w1 = randint(0, actual_pop_size - 1)
                    w2 = randint(0, actual_pop_size - 1)
                    c1 = population[w1].copy()
                    c2 = population[w2].copy()

                    # Random cross-over
                    cross_point = randint(2, 8)
                    temp = list(c1[:cross_point]) + list(c2[cross_point:])
                    new_population.append(temp)

            random_population = self._generate_initial_population(random_population_size)  # noqa: E501

            # Mutation
            for candidate in new_population:
                if uniform(0, 1) < mutation_rate:
                    random_sub_grid = randint(0, 8)
                    possible_swaps = []
                    for grid_element_index in range(self.grid_length):
                        if self.problem_grid[random_sub_grid][grid_element_index] is None:  # noqa: E501
                            possible_swaps.append(grid_element_index)
                    if len(possible_swaps) > 1:
                        shuffle(possible_swaps)
                        first_index = possible_swaps.pop()
                        second_index = possible_swaps.pop()
                        tmp = candidate[random_sub_grid][first_index]
                        candidate[random_sub_grid][first_index] = candidate[random_sub_grid][second_index]  # noqa: E501
                        candidate[random_sub_grid][second_index] = tmp

            new_population = np.append(new_population, random_population, 0)
            population = np.append(population, new_population, 0)
            fitness_history.append(best_fitness)
            if verbose:   # This is used to see progress in console of the solving # noqa: E501
                print(f"Generation {i} -Best candidate's fitness {best_fitness} -Worst candidate's fitness {worst_fitness}")  # noqa: E501

            # Resetting if local minimum
            if len(fitness_history) > 60 and len(np.unique(fitness_history[-49:])) == 1:  # noqa: E501
                print(population[0])
                population = self._generate_initial_population(population_size)  # Reset  # noqa: E501
        (population,
         best_fitness,
         worst_fitness) = self._selection(population, self.selection_rate)
        return population[0], best_fitness, fitness_history
