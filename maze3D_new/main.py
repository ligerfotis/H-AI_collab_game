from config import *
from gameObjects import *
from assets import *
import time

# current layout
layout = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 1, 4, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 1, 4, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 5, 1, 4, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 5, 1, 4, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1, 4, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 5, 1, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 5, 1, 4, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 5, 1, 4, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 5, 1, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 5, 1, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]

board = GameBoard(layout)
keys = {pg.K_UP:1,pg.K_DOWN:2,pg.K_LEFT:4,pg.K_RIGHT:8}
currentKey = 0
running = True
while running:

    for event in pg.event.get():

        if event.type==pg.QUIT:
            running = False
        if event.type==pg.KEYDOWN:
            if event.key == pg.K_SPACE:
                input("Press Enter to continue: ")
            if event.key in keys:
                currentKey += keys[event.key]
        if event.type==pg.KEYUP:
            if event.key in keys:
                currentKey -= keys[event.key]

    board.handleKeys(currentKey)
    board.update()
    glClearDepth(1000.0)
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
    board.draw()
    ballCoords = board.getBallCoords()
    pg.draw.line(screen, (255, 255, 255), (0,0), ballCoords)
    pg.display.flip()
    clock.tick()
    fps = clock.get_fps()
    pg.display.set_caption("Running at "+str(int(fps))+" fps")

    # if train_game_number has ended (goal reached), then reset the board and wait 5 secs to start the next train_game_number
    # while logging the countdown
    if distance.euclidean([board.ball.x, board.ball.y], [board.hole.x, board.hole.y]) < 5:
        timeStart = time.time()
        i=0
        while time.time() - timeStart <= 5:
            board = GameBoard(layout)
            board.update()
            glClearDepth(1000.0)
            glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
            board.draw(mode=True, idx=i)
            pg.display.flip()
            time.sleep(1)
            i+=1


pg.quit()