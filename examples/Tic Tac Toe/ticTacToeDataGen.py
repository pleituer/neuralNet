import numpy as np
import math
import random
import time

from ticTacToeClasses import *

#board and the minimax bot
t = TicTacToe()
bot = Minimax(TRAINSIDE)

#generates all different states of the board (3^8)
def genPer(n, k):
    if n == 1: return [[__] for __ in range(k)]
    pers = []
    for __ in range(k): pers += [[__]+lper for lper in genPer(n-1, k)]
    return pers

#filters out all that the bot will not encounter duing the match
filt1 = lambda b: (0 in b) and (b.count(0)+(0,0,1)[TRAINSIDE] == b.count(2))

rawStarts = [np.array([[c-1 for c in board[t.size*r:t.size*(r+1)]] for r in range(t.size)]) for board in list(filter(filt1, genPer(9, 3)))]
starts = []

for start in rawStarts:
    t.set(board=start)
    t.check()
    if t.status is None: starts.append(start)

#number of data
print('Num of data:', len(starts))

#resets
t.reset()

#generates input and output raw data
Xraw = []
Yraw = []
progress = 0

startTime = time.time()
print("Generating data...")
for start in starts:
    t.set(board=start)
    botMove = bot.getMove(t)
    Xraw.append(t.squish(start))
    Yraw.append((botMove, t.emptySpots()))
    progress += 1
    print(f'Progress: {progress}/{len(starts)}\x1b[1A')
print(f'Progress: {progress}/{len(starts)}')
endTime = time.time()
print(f'Finished generating data, took {endTime - startTime} seconds')

#modifies and saves it
startTime = time.time()
print('Modifying and saving data...')
f = open("ticTacToeTRAINDATA.txt", "w")
f.write(f"D {len(starts)} {t.size**2} {t.size**2}\n")
for i in range(len(starts)):
    f.write("X " + ' '.join(map(str, Xraw[i])) + "\n")
    f.write("Y " + ' '.join(map(str, [int(m == Yraw[i][0][0]*t.size + Yraw[i][0][1])-int((m//t.size, m%t.size) not in Yraw[i][1]) for m in range(t.size**2)])) + "\n")
endTime = time.time()
print(f'Finish modifying and saving data, took {endTime - startTime} seconds')
