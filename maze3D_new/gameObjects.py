from maze3D_new.config import *
from maze3D_new.assets import *
import math
import numpy as np
from scipy.spatial import distance
import time

ball_diameter = 32

class GameBoard:
    def __init__(self, layout, discrete=False, rl=False):
        self.velocity = [0, 0]
        self.walls = []
        self.layout = layout
        self.discrete = discrete
        self.rl = rl
        self.scaling_x = 0.05 if self.discrete else 0.01
        self.scaling_y = 0.01 if self.rl else self.scaling_x

        for row in range(len(layout)):
            self.walls.append([])
            for col in range(len(layout[0])):
                self.walls[row].append(None)
                if layout[row][col] != 0:
                    if layout[row][col] == 2:
                        self.hole = Hole(32 * col - 240, 32 * row - 240, self)
                    elif layout[row][col] == 3:
                        self.ball = Ball(32 * col - 240, 32 * row - 240, self)
                    else:
                        self.walls[row][col] = Wall(32 * col - 240, 32 * row - 240, layout[row][col], self)

        self.rot_x = 0
        self.rot_y = 0
        self.max_x_rotation = 0.5
        self.max_y_rotation = 0.5
        self.count_slide = 0
        self.slide = False
        self.slide_velx, self.slide_vely = 0, 0

        self.keyMap = {1: (1, 0),
                       2: (-1, 0),
                       4: (0, 1), 5: (1, 1), 6: (-1, 1), 7: (0, 1),
                       8: (0, -1), 9: (1, -1), 10: (-1, -1), 11: (0, -1), 13: (1, 0), 14: (-1, 0)}

    def getBallCoords(self):
        return (self.ball.x, self.ball.y)

    def collideSquare(self, x, y):
        # if the ball hits a square obstacle, it will return True
        # and the collideTriangle will not be called

        xGrid = math.floor((x + 240) / 32)
        yGrid = math.floor((y + 240) / 32)

        biggest = max(xGrid, yGrid)
        smallest = min(xGrid, yGrid)
        if biggest > 13 or smallest < 0:
            return True, None
        if self.walls[yGrid][xGrid] is not None:
                return True, self.layout[yGrid][xGrid]
        return False, None

    def collideTriangle(self, checkX, checkY, x, y, velx, vely, accx, accy):
        # find the grid that the ball tends to enter
        # grid_directionX stores the coordinates of the ball in the x axis
        # grid_directionY stores the coordinates of the ball in the y axis

        grid_directionX = [math.floor((checkX + 240) / 32), math.floor((y + 240) / 32)]
        grid_directionY = [math.floor((x + 240) / 32), math.floor((checkY + 240) / 32)]

        check_collision = [grid_directionX, grid_directionY]

        # check that the ball moves in the obstacle-free space
        for direction in check_collision:

            # if the grid has not an object
            if self.layout[direction[1]][direction[0]] in [0, 3] and not self.slide:
                return velx, vely, False

        # if code reaches this point, we are at a grid of triangle obstacle

        # change reference point to be down left pixel of the grid
        xObs, yObs = 0, 0
        xBall, yBall = x - 32 * direction[0] + 240, y - 32 * direction[1] + 240

        # get the point of the ball that will hit the triangle obstacle
        xCol4, yCol4 = x + 8 * np.cos(225 * np.pi / 180), y + 8 * np.sin(225 * np.pi / 180)
        xGridCol4, yGridCol4 = math.floor((xCol4 + 240) / 32), math.floor((yCol4 + 240) / 32)
        xCol5, yCol5 = x + 8 * np.cos(45 * np.pi / 180), y + 8 * np.sin(45 * np.pi / 180)
        xGridCol5, yGridCol5 = math.floor((xCol5 + 240) / 32), math.floor((yCol5 + 240) / 32)

        if self.layout[yGridCol4][xGridCol4] == 4 or self.layout[yGridCol4][xGridCol4] == 6:

            # collision angle
            theta = 225 * np.pi / 180
            xCol, yCol = xBall + 8 * np.cos(theta), yBall + 8 * np.sin(theta)
            thetaCol = np.arctan((yCol - yObs) / (xCol - 32 - xObs)) * 180 / np.pi

            # print(self.layout[yGridCol4][xGridCol4])
            # print(thetaCol)
            if self.layout[yGridCol4][xGridCol4] == 6:
                thetaCol = 135
            else:
                thetaCol += 180

            # if thetaCol is less than 135 degrees, reset the slide counter and the slide flag
            # and return the commanded velocities
            if thetaCol < 135:
                self.count_slide = 0
                self.slide = False
                self.slide_velx, self.slide_vely = 0, 0
                return velx, vely, False
            # if thetaCol is greater than 135 degrees, then the ball hit the triangle
            elif thetaCol >= 135:
                # if collision angle is greater than 135 degrees 3 consecutive times, 
                # then we assume that the ball touches the leaning surface. Otherwise, the ball
                # will bounce 
                if self.count_slide == 3:
                    self.slide = True
                elif not self.slide:
                    self.slide = False
                    self.count_slide += 1
                    if velx <= 0 and vely <= 0:
                        return 0.25 * abs(vely), 0.25 * abs(velx), False
                    return 0.25 * np.sign(velx) * abs(vely), 0.25 * np.sign(vely) * abs(velx), False

                if self.slide:
                    if accx > 0 and accy < 0:
                        self.slide_velx = self.slide_velx + accx + abs(accy) * np.sin(np.pi / 4) ** 2
                        self.slide_vely += accy * np.sin(np.pi / 4) ** 2
                    elif accx < 0 and accy > 0:
                        self.slide_velx += accx * np.sin(np.pi / 4) ** 2
                        self.slide_vely = self.slide_vely + accy + abs(accx) * np.sin(np.pi / 4) ** 2
                    elif accx > 0 and accy > 0:
                        self.slide_velx += accx
                        self.slide_vely += accy
                    else:
                        self.slide_velx = self.slide_velx + abs(accy) * np.sin(np.pi / 4) ** 2 + accx * np.sin(
                            np.pi / 4) ** 2
                        self.slide_vely = self.slide_vely + abs(accx) * np.sin(np.pi / 4) ** 2 + accy * np.sin(
                            np.pi / 4) ** 2
                    return self.slide_velx, self.slide_vely, False


        # right triangle
        elif self.layout[yGridCol5][xGridCol5] == 5 or self.layout[yGridCol5][xGridCol5] == 6:
            # collision angle
            theta = 45 * np.pi / 180
            xCol, yCol = xBall + 8 * np.cos(theta), yBall + 8 * np.sin(theta)
            thetaCol = np.arctan((yCol - yObs) / (xCol - 32 - xObs)) * 180 / np.pi

            if self.layout[yGridCol5][xGridCol5] == 6:
                thetaCol = 135
            else:
                thetaCol += 180

            # if thetaCol is greater than 135 degrees, reset the slide counter and the slide flag
            # and return the commanded velocities
            if thetaCol > 135:
                self.count_slide = 0
                self.slide = False
                self.slide_velx, self.slide_vely = 0, 0
                return velx, vely, False
            # if thetaCol is less than 135 degrees, then the ball hit the triangle
            elif thetaCol <= 135:
                self.count_slide += 1
                # if collision angle is greater than 135 degrees 3 consecutive times, 
                # then we assume that the ball touches the leaning surface.
                # Otherwise, the ball will bounce 
                if self.count_slide == 3:
                    self.slide = True
                elif not self.slide:
                    if velx >= 0 and vely >= 0:
                        return -0.25 * vely, -0.25 * velx, False
                    return 0.25 * np.sign(velx) * abs(vely), 0.25 * np.sign(vely) * abs(velx), False
                if self.slide:
                    if accx > 0 and accy < 0:
                        self.slide_velx += accx * np.sin(np.pi / 4) ** 2
                        self.slide_vely = self.slide_vely + accy - accx * np.sin(np.pi / 4) ** 2
                    elif accx < 0 and accy > 0:
                        self.slide_velx = self.slide_velx + accx - accy * np.sin(np.pi / 4) ** 2
                        self.slide_vely += accy * np.sin(np.pi / 4) ** 2
                    elif accx < 0 and accy < 0:
                        self.slide_velx += accx
                        self.slide_vely += accy
                    else:
                        self.slide_velx = self.slide_velx - accy * np.sin(np.pi / 4) ** 2 + accx * np.sin(
                            np.pi / 4) ** 2
                        self.slide_vely = self.slide_vely - accx * np.sin(np.pi / 4) ** 2 + accy * np.sin(
                            np.pi / 4) ** 2

                    return self.slide_velx, self.slide_vely, False
        return velx, vely, False

    def update(self):
        # compute rotation matrix
        rot_x_m = pyrr.Matrix44.from_x_rotation(self.rot_x)
        rot_y_m = pyrr.Matrix44.from_y_rotation(self.rot_y)
        self.rotationMatrix = pyrr.matrix44.multiply(rot_x_m, rot_y_m)

        self.ball.update()
        self.hole.update()

        for row in self.walls:
            for wall in row:
                if wall != None:
                    wall.update()

    def handleKeys(self, angleIncrement):
        if angleIncrement[0] == 2:
            angleIncrement[0] = -1
        elif angleIncrement[0] == 1:
            angleIncrement[0] = 1

        if angleIncrement[1] == 2:
            angleIncrement[1] = -1
        elif angleIncrement[1] == 1:
            angleIncrement[1] = 1

        self.velocity[0] = self.scaling_y * angleIncrement[0]
        self.rot_x += self.velocity[0]
        if self.rot_x >= self.max_x_rotation:
            self.rot_x = self.max_x_rotation
            self.velocity[0] = 0
        elif self.rot_x <= -self.max_x_rotation:
            self.rot_x = -self.max_x_rotation
            self.velocity[0] = 0

        self.velocity[1] = self.scaling_x * angleIncrement[1]
        self.rot_y += self.velocity[1]
        if self.rot_y >= self.max_y_rotation:
            self.rot_y = self.max_y_rotation
            self.velocity[1] = 0
        elif self.rot_y <= -self.max_y_rotation:
            self.rot_y = -self.max_y_rotation
            self.velocity[1] = 0

    def draw(self, mode=0, idx=0):
        translation = pyrr.matrix44.create_from_translation(pyrr.Vector3([-80, -80, 0]))
        self.model = pyrr.matrix44.multiply(translation, self.rotationMatrix)
        glUniformMatrix4fv(MODEL_LOC, 1, GL_FALSE, self.model)
        glBindVertexArray(BOARD_MODEL.getVAO())
        glBindTexture(GL_TEXTURE_2D, BOARD.getTexture())
        glDrawArrays(GL_TRIANGLES, 0, BOARD_MODEL.getVertexCount())

        self.ball.draw()
        self.hole.draw()

        for row in self.walls:
            for wall in row:
                if wall != None:
                    wall.draw()
        # Used for resetting the game. Logs above the board "Game starts in ..."
        if mode == 1:
            translation = pyrr.matrix44.create_from_translation(pyrr.Vector3([-60, 350, 0]))
            glUniformMatrix4fv(MODEL_LOC, 1, GL_FALSE,
                               pyrr.matrix44.multiply(translation, pyrr.matrix44.create_identity()))
            glBindVertexArray(TEXT_MODEL.getVAO())
            glBindTexture(GL_TEXTURE_2D, TEXT[idx].getTexture())
            glDrawArrays(GL_TRIANGLES, 0, TEXT_MODEL.getVertexCount())
        # Used when goal has been reached. Logs above the board "Goal reached"
        elif mode == 2:
            translation = pyrr.matrix44.create_from_translation(pyrr.Vector3([-60, 350, 0]))
            glUniformMatrix4fv(MODEL_LOC, 1, GL_FALSE,
                               pyrr.matrix44.multiply(translation, pyrr.matrix44.create_identity()))
            glBindVertexArray(TEXT_MODEL.getVAO())
            glBindTexture(GL_TEXTURE_2D, TEXT[-2].getTexture())
            glDrawArrays(GL_TRIANGLES, 0, TEXT_MODEL.getVertexCount())
        # Used for resetting the game. Logs above the board "Timeout"
        elif mode == 3:
            translation = pyrr.matrix44.create_from_translation(pyrr.Vector3([-60, 350, 0]))
            glUniformMatrix4fv(MODEL_LOC, 1, GL_FALSE,
                               pyrr.matrix44.multiply(translation, pyrr.matrix44.create_identity()))
            glBindVertexArray(TEXT_MODEL.getVAO())
            glBindTexture(GL_TEXTURE_2D, TEXT[-1].getTexture())
            glDrawArrays(GL_TRIANGLES, 0, TEXT_MODEL.getVertexCount())


class Wall:
    def __init__(self, x, y, type, parent):
        self.parent = parent
        self.x = x
        self.y = y
        self.z = 0
        if type in [6, 7]:
            type = 1
        self.type = type - 1

    def update(self):
        # first translate to position on board, then rotate with the board
        translation = pyrr.matrix44.create_from_translation(pyrr.Vector3([self.x, self.y, self.z]))
        self.model = pyrr.matrix44.multiply(translation, self.parent.rotationMatrix)

    def draw(self):
        glUniformMatrix4fv(MODEL_LOC, 1, GL_FALSE, self.model)
        glBindVertexArray(WALL_MODELS[self.type].getVAO())
        glBindTexture(GL_TEXTURE_2D, WALL.getTexture())
        glDrawArrays(GL_TRIANGLES, 0, WALL_MODELS[self.type].getVertexCount())


class Ball:
    def __init__(self, x, y, parent):
        self.parent = parent
        self.x = x
        self.y = y
        self.z = 0
        self.velocity = [0, 0]

    def update(self):
        # first translate to position on board, then rotate with the board
        translation = pyrr.matrix44.create_from_translation(pyrr.Vector3([self.x, self.y, self.z]))
        self.model = pyrr.matrix44.multiply(translation, self.parent.rotationMatrix)

        acceleration = [-0.1 * self.parent.rot_y, 0.1 * self.parent.rot_x]
        self.velocity[0] += 0.5 * acceleration[0]
        self.velocity[1] += 0.5 * acceleration[1]

        nextX = self.x + self.velocity[0]
        nextY = self.y + self.velocity[1]

        test_nextX = nextX + ball_diameter/2 * np.sign(self.velocity[0])
        test_nextY = nextY + ball_diameter/2 * np.sign(self.velocity[1])

        # check x direction
        checkXCol, gridX = self.parent.collideSquare(test_nextX, self.y)
        checkYCol, gridY = self.parent.collideSquare(self.x, test_nextY)
        # checkXYCol, gridXY = self.parent.collideSquare(test_nextX, test_nextY)

        if checkXCol:
            self.velocity[0] *= -0.25

        # check y direction
        if checkYCol:
            self.velocity[1] *= -0.25

        # # print (self.velocity)
        # if (not checkXCol and not checkYCol) or gridX == 7 or gridY == 7:
        #     velx, vely, collision = self.parent.collideTriangle(test_nextX, test_nextY, nextX, nextY,
        #                                                         self.velocity[0], self.velocity[1], acceleration[0],
        #                                                         acceleration[1])
        #     # print(velx, vely)
        #     if collision:
        #         self.velocity[0] *= -0.25
        #         self.velocity[1] *= -0.25
        #     else:
        #         self.velocity = [velx, vely]

        self.x += self.velocity[0]
        self.y += self.velocity[1]

    def draw(self):
        glUniformMatrix4fv(MODEL_LOC, 1, GL_FALSE, self.model)
        glBindVertexArray(BALL_MODEL.getVAO())
        glBindTexture(GL_TEXTURE_2D, BALL.getTexture())
        glDrawArrays(GL_TRIANGLES, 0, BALL_MODEL.getVertexCount())


class Hole:
    def __init__(self, x, y, parent):
        self.parent = parent
        self.x = x
        self.y = y
        self.z = 0

    def update(self):
        # first translate to position on board, then rotate with the board
        translation = pyrr.matrix44.create_from_translation(pyrr.Vector3([self.x, self.y, self.z]))
        self.model = pyrr.matrix44.multiply(translation, self.parent.rotationMatrix)

    def draw(self):
        glUniformMatrix4fv(MODEL_LOC, 1, GL_FALSE, self.model)
        glBindVertexArray(HOLE_MODEL.getVAO())
        glBindTexture(GL_TEXTURE_2D, HOLE.getTexture())
        glDrawArrays(GL_TRIANGLES, 0, HOLE_MODEL.getVertexCount())
