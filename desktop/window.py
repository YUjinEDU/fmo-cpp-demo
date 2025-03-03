import cv2
import numpy as np
import time
import threading
from collections import deque

class Colour:
    def __init__(self, b, g, r):
        self.b = b
        self.g = g
        self.r = r

    @staticmethod
    def red():
        return Colour(0x40, 0x40, 0x90)

    @staticmethod
    def green():
        return Colour(0x40, 0x80, 0x40)

    @staticmethod
    def blue():
        return Colour(0xA0, 0x40, 0x40)

    @staticmethod
    def magenta():
        return Colour(0xA0, 0x40, 0x90)

    @staticmethod
    def gray():
        return Colour(0x60, 0x60, 0x60)

    @staticmethod
    def lightRed():
        return Colour(0x40, 0x40, 0xFF)

    @staticmethod
    def lightGreen():
        return Colour(0x40, 0xFF, 0x40)

    @staticmethod
    def lightBlue():
        return Colour(0xFF, 0x40, 0x40)

    @staticmethod
    def lightMagenta():
        return Colour(0xFF, 0x40, 0xFF)

    @staticmethod
    def lightGray():
        return Colour(0xC0, 0xC0, 0xC0)

class Window:
    def __init__(self):
        self.mFrameNs = 0
        self.mLastNs = 0
        self.mLines = []
        self.mLineClrs = []
        self.mColour = Colour.lightGray()
        self.mOpen = False
        self.mBottomLine = ""
        self.mTopLine = ""
        self.mCenterLine = ""
        self.mCenterUnderLine = ""
        self.mTable = []
        self.visTable = False

    def close(self):
        if not self.mOpen:
            return
        self.mOpen = False
        cv2.destroyWindow("FMO")

    def setTextColor(self, color):
        self.mColour = color

    def print(self, line, clr=None):
        self.mLines.append(line)
        self.mLineClrs.append(clr if clr else self.mColour)

    def setBottomLine(self, text):
        self.mBottomLine = text

    def setCenterLine(self, text, textunder):
        self.mCenterLine = text
        self.mCenterUnderLine = textunder

    def setTopLine(self, text):
        self.mTopLine = text

    def display(self, image):
        self.open(image.shape[:2])
        mat = image
        self.printText(mat)
        cv2.imshow("FMO", mat)
        cv2.setWindowProperty("FMO", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def setFrameTime(self, sec):
        self.mFrameNs = int(1e9 * sec)

    def getCommand(self, block):
        keyCode = cv2.waitKey(0 if block else 1)
        sinceLastNs = time.time_ns() - self.mLastNs
        waitNs = self.mFrameNs - sinceLastNs - 1_000_000

        if waitNs > 0:
            time.sleep(waitNs / 1e9)

        self.mLastNs = time.time_ns()
        return self.encodeKey(keyCode)

    def open(self, dims):
        if self.mOpen:
            return
        self.mOpen = True
        cv2.namedWindow("FMO", cv2.WINDOW_NORMAL)

        if dims[0] > 800:
            dims = (dims[1] // 2, dims[0] // 2)

        cv2.resizeWindow("FMO", dims[1], dims[0])

    def encodeKey(self, keyCode):
        key_map = {
            27: "QUIT",
            13: "STEP",
            10: "STEP",
            ord(' '): "PAUSE",
            ord('f'): "PAUSE_FIRST",
            ord(','): "JUMP_BACKWARD",
            ord('.'): "JUMP_FORWARD",
            ord('n'): "INPUT",
            ord('r'): "RECORD",
            ord('R'): "RECORD",
            ord('g'): "RECORD_GRAPHICS",
            ord('G'): "RECORD_GRAPHICS",
            ord('a'): "AUTOMATIC_MODE",
            ord('A'): "AUTOMATIC_MODE",
            ord('m'): "MANUAL_MODE",
            ord('M'): "MANUAL_MODE",
            ord('e'): "FORCED_EVENT",
            ord('E'): "FORCED_EVENT",
            ord('?'): "SHOW_HELP",
            ord('h'): "SHOW_HELP",
            ord('H'): "SHOW_HELP",
            ord('s'): "PLAY_SOUNDS",
            ord('S'): "PLAY_SOUNDS",
            ord('p'): "SCREENSHOT",
            ord('P'): "SCREENSHOT",
            ord('0'): "LEVEL0",
            ord('1'): "LEVEL1",
            ord('2'): "LEVEL2",
            ord('3'): "LEVEL3",
            ord('4'): "LEVEL4",
            ord('5'): "LEVEL5",
            ord('l'): "LOCAL_MAXIMA",
            ord('d'): "DIFF",
            ord('b'): "BIN_DIFF",
            ord('i'): "SHOW_IM",
            ord('t'): "DIST_TRAN",
            ord('o'): "SHOW_NONE"
        }
        return key_map.get(keyCode, "NONE")

    def printText(self, mat):
        fontFace = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = mat.shape[0] / 1000.0
        thick = 2 if mat.shape[0] > 800 else 1
        pad = int(fontScale * 10)
        color = (self.mColour.b, self.mColour.g, self.mColour.r)

        lineHeight = cv2.getTextSize("ABC", fontFace, fontScale, thick)[0][1]
        above = (9 * lineHeight // 14) + (lineHeight // 2)
        below = 5 * lineHeight // 14

        if self.mLines:
            lineWidth = max(cv2.getTextSize(line, fontFace, fontScale, thick)[0][0] for line in self.mLines)
            xMax = 2 * pad + lineWidth
            yMax = 2 * pad + len(self.mLines) * (above + below)
            mat[:yMax, :xMax] = (0.3 * mat[:yMax, :xMax]).astype(np.uint8)

            y = pad
            for line, clr in zip(self.mLines, self.mLineClrs):
                y += above
                cv2.putText(mat, line, (pad, y), fontFace, fontScale, (clr.b, clr.g, clr.r), thick)
                y += below

            self.mLines.clear()
            self.mLineClrs.clear()

        if self.mBottomLine:
            helpRectHeight = (above + below) + 2 * pad
            mat[-helpRectHeight:, :] = (0.3 * mat[-helpRectHeight:, :]).astype(np.uint8)
            cv2.putText(mat, self.mBottomLine, (pad, mat.shape[0] - pad - below), fontFace, fontScale, color, thick)

        if self.mTopLine:
            lineSize = cv2.getTextSize(self.mTopLine, fontFace, fontScale, thick)[0]
            helpRectHeight = (above + below) + 2 * pad
            lineWidth = lineSize[0] + 2 * pad
            offset = (mat.shape[1] - lineWidth) // 2
            mat[:helpRectHeight, offset:offset + lineWidth] = (0.3 * mat[:helpRectHeight, offset:offset + lineWidth]).astype(np.uint8)
            cv2.putText(mat, self.mTopLine, (pad + offset, above + pad // 2), fontFace, fontScale, color, thick)

        if self.mCenterLine or self.mCenterUnderLine:
            fontScaleCenter = 3 * fontScale
            thickCenter = 3 * thick
            lineSize = cv2.getTextSize(self.mCenterLine, fontFace, fontScaleCenter, thickCenter)[0]
            lineWidth = lineSize[0] + 2 * pad
            lineHeight = lineSize[1] + 2 * pad
            offsetW = (mat.shape[1] - lineWidth) // 2 + 4 * pad
            offsetH = (mat.shape[0] - lineHeight) // 2 + 2 * lineHeight
            mat[offsetH:offsetH + lineHeight, offsetW:offsetW + lineWidth] = 0
            cv2.putText(mat, self.mCenterLine, (offsetW + pad, offsetH - 2 * pad), fontFace, fontScaleCenter, color, thickCenter)
            cv2.putText(mat, self.mCenterUnderLine, (offsetW + pad, offsetH + lineHeight - 2 * pad), fontFace, fontScaleCenter, color, thickCenter)

        if self.visTable and self.mTable:
            lineSize1 = cv2.getTextSize("10. ", fontFace, fontScale, thick)[0]
            lineSize2 = cv2.getTextSize("wwwwwwwwww  ", fontFace, fontScale, thick)[0]
            lineSize3 = cv2.getTextSize("444.44", fontFace, fontScale, thick)[0]
            lineWidth = lineSize1[0] + lineSize2[0] + lineSize3[0]

            xMax = pad + lineWidth
            yMax = 2 * pad + len(self.mTable) * (above + below)
            mat[:yMax, :xMax] = (0.3 * mat[:yMax, :xMax]).astype(np.uint8)

            y = pad
            for i, (score, name) in enumerate(self.mTable):
                pref = " " if i < 9 else ""
                y += above
                cv2.putText(mat, f"{pref}{i + 1}.", (pad, y), fontFace, fontScale, color, thick)
                cv2.putText(mat, name, (lineSize1[0] + pad, y), fontFace, fontScale, color, thick)
                cv2.putText(mat, f"{score:.2f}", (lineSize1[0] + lineSize2[0] + pad, y), fontFace, fontScale, color, thick)
                y += below

def drawPoints(points, target, colour):
    mat = target
    vec = (colour.b, colour.g, colour.r)
    for pt in points:
        mat[pt[1], pt[0]] = vec

def drawPointsGt(ps, gt, target):
    mat = target
    c1 = (Colour.lightMagenta().b, Colour.lightMagenta().g, Colour.lightMagenta().r)
    c2 = (Colour.lightRed().b, Colour.lightRed().g, Colour.lightRed().r)
    c3 = (Colour.lightGreen().b, Colour.lightGreen().g, Colour.lightGreen().r)
    for pt in ps:
        if pt in gt:
            mat[pt[1], pt[0]] = c3
        else:
            mat[pt[1], pt[0]] = c1
    for pt in gt:
        if pt not in ps:
            mat[pt[1], pt[0]] = c2

def removePoints(points, target, bg):
    mat = target
    b = bg
    for pt in points:
        mat[pt[1], pt[0]] = b[pt[1], pt[0]]
