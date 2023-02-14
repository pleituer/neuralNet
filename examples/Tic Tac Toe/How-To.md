# Tic Tac Toe AI

## How to use

Note that `O` means going second, `X` means going first

You may set whether the bot should go first or second by setting the variable `TRAINSIDE` under the file `ticTacToeClasses.py` to be `X` or `O` respectively.

Run the following command to generate data, making use of minimax algorithm
```
~$ python3 ticTacToeDataGen.py
```
Then run the following command to train and play with the nerual network AI
```
~$ python3 ticTacToe.py
```
Or you can just play with the minimax algorithm by the command
```
~$ python3 ticTacToePlay.py
```

When interacting, the format for input is: `> {column} {row}`

For example, entering `2 1` will mean your desired spot is column 2, row 1. Both row and column starts counting from 0, so `0 0` will mean the top left corner, `2 0` will mean the top right corner.
