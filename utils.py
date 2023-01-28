import cv2 as cv
import numpy as np
import logging
import cvzone

MAVI = (255,0,0)
YESIL = (0,255,0)
KIRMIZI = (0,0,255)


def preprocess(img):
    imgGray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    imgBlur = cv.GaussianBlur(imgGray,(5,5),0)
    imgCanny = cv.Canny(imgBlur,218,103)
    return imgCanny


def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv.cvtColor( imgArray[x][y], cv.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv.cvtColor(imgArray[x], cv.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver


def findContours(imgForDraw, imgForFind, minArea=100, sort=True, drawContours=True, drawRectangles=True, drawCenters=True,c=MAVI,cornerFilter="none"):
    allContours = []
    contours, hierarchy = cv.findContours(imgForFind, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    imgContoured = imgForDraw.copy()

    for cnt in contours:
        okay = False
        area = cv.contourArea(cnt)
        if area > minArea:
            peri = cv.arcLength(cnt, True)
            cornerPoints = cv.approxPolyDP(cnt, 0.02 * peri, True)
            cornerCount = len(cornerPoints)
            match cornerFilter:
                case "daire":
                    if cornerCount > 5:
                        okay = True
                case "dortgen":
                    if cornerCount == 4:
                        okay = True
                case "ucgen":
                    if cornerCount == 3:
                        okay = True
                case _:
                    okay = True

            if okay:
                x, y, w, h = cv.boundingRect(cornerPoints)
                cx, cy = x + (w // 2), y + (h // 2)
                if drawContours: 
                    cv.drawContours(imgContoured, cnt, -1, c, 3)
                if drawRectangles:
                    cv.rectangle(imgContoured, (x, y), (x + w, y + h), c, 2)
                if drawCenters:
                    cv.circle(imgContoured, (x + (w // 2), y + (h // 2)), 5, c, cv.FILLED)
                allContours.append({"contourPoint": cnt, "area": area, "bbox": [x, y, w, h], "center": [cx, cy]})

    if sort:
        allContours = sorted(allContours, key=lambda x: x["area"], reverse=True)

    return imgContoured, allContours


def findCorners(allContours,imgForDraw,drawCorners=False,color=MAVI,cornerFilter="none"):
    allCorners = []
    if allContours:
        for contour in allContours:
            peri = cv.arcLength(contour["contourPoint"],True)
            cornerPoints = cv.approxPolyDP(contour["contourPoint"],0.02*peri,True)
            cornerCount = len(cornerPoints)

            match cornerFilter:
                case "daire":
                    if cornerCount > 4:
                        allCorners.append({"coordinates":cornerPoints, "cornerCount":cornerCount})
                case "dortgen":
                    if cornerCount == 4:
                        allCorners.append({"coordinates":cornerPoints, "cornerCount":cornerCount})
                case "ucgen":
                    if cornerCount == 3:
                        allCorners.append({"coordinates":cornerPoints, "cornerCount":cornerCount})
                case _:
                    allCorners.append({"coordinates":cornerPoints, "cornerCount":cornerCount})

    if drawCorners:
        for corner in allCorners:
            cv.drawContours(imgForDraw,corner["coordinates"],-1,color,12)
    return imgForDraw, allCorners
 

def overlayPNG(imgBack, imgFront, pos=[0, 0]):
    hf, wf, cf = imgFront.shape
    hb, wb, cb = imgBack.shape
    *_, mask = cv.split(imgFront)
    maskBGRA = cv.cvtColor(mask, cv.COLOR_GRAY2BGRA)
    maskBGR = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
    imgRGBA = cv.bitwise_and(imgFront, maskBGRA)
    imgRGB = cv.cvtColor(imgRGBA, cv.COLOR_BGRA2BGR)

    imgMaskFull = np.zeros((hb, wb, cb), np.uint8)
    imgMaskFull[pos[1]:hf + pos[1], pos[0]:wf + pos[0], :] = imgRGB
    imgMaskFull2 = np.ones((hb, wb, cb), np.uint8) * 255
    maskBGRInv = cv.bitwise_not(maskBGR)
    imgMaskFull2[pos[1]:hf + pos[1], pos[0]:wf + pos[0], :] = maskBGRInv

    imgBack = cv.bitwise_and(imgBack, imgMaskFull2)
    imgBack = cv.bitwise_or(imgBack, imgMaskFull)

    return imgBack