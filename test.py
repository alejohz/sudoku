import sudoku
import timeit

s = [[0, 0, 4, 3, 0, 0, 2, 0, 9],
     [0, 0, 5, 0, 0, 9, 0, 0, 1],
     [0, 7, 0, 0, 6, 0, 0, 4, 3],
     [0, 0, 6, 0, 0, 2, 0, 8, 7],
     [1, 9, 0, 0, 0, 7, 4, 0, 0],
     [0, 5, 0, 0, 8, 3, 0, 0, 0],
     [6, 0, 0, 0, 0, 0, 1, 0, 5],
     [0, 0, 3, 5, 0, 8, 6, 9, 0],
     [0, 4, 2, 9, 1, 0, 3, 0, 0]]
start = timeit.default_timer()
solver = sudoku.Sudoku(s)
sol, ftns, ftns_history = solver.solve(cross_over="random candidate")
print(ftns)
print(timeit.default_timer() - start)

start = timeit.default_timer()
solver = sudoku.Sudoku(s)
sol, ftns, ftns_history = solver.solve(cross_over="best candidate")
print(ftns)
print(timeit.default_timer() - start)
