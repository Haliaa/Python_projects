import random
from math import exp
from copy import deepcopy

n = int(input("Введіть розмір шахової дошки:"))
temp = 1
ANSI_YELLOW_BACKGROUND = "\u001B[43m"
ANSI_RESET = "\u001B[0m"
ANSI_BLACK = "\u001B[30m"
def ChessBoard(n):
    vector = list(range(n))
    random.shuffle(vector)
    print(vector)
    board = {}
    for _ in range(n): board[_] = vector[_]
    chessboardPrint(board)
    return board

def countAmoutOfThreats(n):
    return n * (n - 1) / 2

def placing(board):
    board_1 = {}
    board_2 = {}
    for i in board:
        counter_1 = i - board[i]
        if counter_1 not in board_1:
            board_1[counter_1] = 1
        else:
            board_1[counter_1] += 1
    for j in board:
        counter_2 = j + board[j]
        if counter_2 not in board_2:
            board_2[counter_2] = 1
        else:
            board_2[counter_2] += 1
    amount = 0
    for k in board_1:
        amount += countAmoutOfThreats(board_1[k])
    for k in board_2:
        amount += countAmoutOfThreats(board_2[k])
    return amount

def annealing_algorithm():
    chessBoard = ChessBoard(n)
    chessThreats = placing(chessBoard)

    t = temp
    tempIncrease = 0.99
    while t > 0:
        t *= tempIncrease
        bestSolution = deepcopy(chessBoard)
        i=0
        j=0
        while (i == j):
            i = random.randrange(0, n-1)
            j = random.randrange(0, n-1)
        bestSolution[i], bestSolution[j] = bestSolution[j], bestSolution[i]
        if placing(bestSolution) - chessThreats < 0 or random.uniform(0, 1) < exp(-(placing(bestSolution) - chessThreats) / t):
            chessBoard = deepcopy(bestSolution)
            chessThreats = placing(chessBoard)

        if chessThreats == 0:
            chessboardPrint(chessBoard)
            break

def chessboardPrint(board):
    print("{0}x{0} Chess board final output:".format(n))
    for k in board.values():
        print(ANSI_YELLOW_BACKGROUND + ANSI_BLACK + "□  " * k + "♕  " + "□  " * (n - k - 1) + ANSI_RESET)

annealing_algorithm()