import cv2 as cv
import numpy as np
import pytesseract as pyts
from utils import *
from sudoku import Sudoku 


# DEĞİŞKENLER
imgW = 450
imgH = 450
pyts.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
cong = '--oem 3 --psm 6 outputbase digits -c tessedit_char_whitelist=0123456789'

img = cv.imread("res/sudoku.png")
img = cv.resize(img,(imgW,imgH))
imgCanny = preprocess(img)
imgBlank = np.zeros_like(img)




# KARELERİ BULMA
imgContoured, allContours = findContours(img, imgCanny,100,True,drawRectangles=False, drawCenters=False, cornerFilter="dortgen")
imgContoured, allCorners = findCorners(allContours,imgContoured,drawCorners=True,cornerFilter="dortgen",color=YESIL)


if len(allCorners) != 0:
    sudokuCorners = allCorners[0]["coordinates"]
    sudokuCorners = np.reshape(np.asarray(sudokuCorners),(4,2))


    # PERSPEKTİF DÜZELTME
    pointsBefore = np.float32(sudokuCorners)
    pointsAfter = np.float32([[imgW,0],[0,0],[0,imgH],[imgW,imgH]])

    matrix = cv.getPerspectiveTransform(pointsBefore,pointsAfter)
    sudoku = cv.warpPerspective(img,matrix,(imgW,imgH))
    sudoku = np.asarray(sudoku)

    sudokuGray = cv.cvtColor(sudoku,cv.COLOR_BGR2GRAY)
    sudokuThres = cv.threshold(sudokuGray,150,255,cv.THRESH_BINARY_INV)[1]


    # SUDOKOYU KARELERE AYIRMA
    imgSudokuSolved = sudoku.copy()
    boxes = []

    azaltma = 4

    rows = np.vsplit(sudoku,9)
    for row in rows:
        columns = np.hsplit(row,9)
        for box in columns:
            boxes.append(box)

    for i in range(0,81):
        boxes[i] = boxes[i][azaltma:50-azaltma,azaltma:50-azaltma]
        boxGray = cv.cvtColor(boxes[i],cv.COLOR_BGR2GRAY)
        boxThres = cv.threshold(boxGray,150,255,cv.THRESH_BINARY_INV)[1] 
        boxPixel = cv.countNonZero(boxThres)
        print(boxPixel)
        if boxPixel < 10:
            cv.putText(boxes[i],"0",(11,32),cv.FONT_HERSHEY_COMPLEX,1,MAVI,2)

    
    sudokuNew = stackImages(0.6,
    [[boxes[0], boxes[1], boxes[2], boxes[3],boxes[4],boxes[5],boxes[6],boxes[7],boxes[8]],
    [boxes[9], boxes[10], boxes[11], boxes[12],boxes[13],boxes[14],boxes[15],boxes[16],boxes[17]],
    [boxes[18], boxes[19], boxes[20], boxes[21],boxes[22],boxes[23],boxes[24],boxes[25],boxes[26]],
    [boxes[27], boxes[28], boxes[29], boxes[30],boxes[31],boxes[32],boxes[33],boxes[34],boxes[35]],
    [boxes[36], boxes[37], boxes[38], boxes[39],boxes[40],boxes[41],boxes[42],boxes[43],boxes[44]],
    [boxes[45], boxes[46], boxes[47], boxes[48],boxes[49],boxes[50],boxes[51],boxes[52],boxes[53]],
    [boxes[54], boxes[55], boxes[56], boxes[57],boxes[58],boxes[59],boxes[60],boxes[61],boxes[62]],
    [boxes[63], boxes[64], boxes[65], boxes[66],boxes[67],boxes[68],boxes[69],boxes[70],boxes[71]],
    [boxes[72], boxes[73], boxes[74], boxes[75],boxes[76],boxes[77],boxes[78],boxes[79],boxes[80]]])
    
    sudokuNew = cv.resize(sudokuNew,(imgW,imgH))
    sudokuNewRGB =cv.cvtColor(sudokuNew, cv.COLOR_BGR2RGB)


    # SAYILARI BULMA
    sudokuList = [[],[],[],[],[],[],[],[],[]]
    boxList = []

    allDigits = pyts.image_to_boxes(sudokuNewRGB, config=cong)
    print(allDigits)
    for line in allDigits.splitlines():
        box = line.split(" ")
        boxList.append(box)

    for i,box in enumerate(boxList):
        text = box[0]
        x = int(box[1])
        y = int(box[2])
        w = int(box[3])
        h = int(box[4])
        cv.rectangle(sudokuNew, (x, imgH-y), (w, imgH-h), MAVI, 1)
        x1 = i // 9
        row = sudokuList[x1]
        row.append(int(text))

    print(sudokuList)

    puzzle = Sudoku(3, 3, board=sudokuList)
    solution = puzzle.solve()
    solution.show()

    sudokuListSolved = solution.board
    sudokuBlank = np.zeros_like(img)

    kutular = []

    azaltma = 4

    rowss = np.vsplit(sudokuBlank,9)
    for row in rowss:
        columns = np.hsplit(row,9)
        for box in columns:
            kutular.append(box)

    for i in range(0,81):
        x = i % 9
        y = i // 9
        if sudokuList[y][x] == 0:
            cv.putText(kutular[i],str(sudokuListSolved[y][x]),(11,37),cv.FONT_HERSHEY_DUPLEX,1,YESIL,1)


    sudokuNewSolved = stackImages(0.6,[[kutular[0], kutular[1], kutular[2], kutular[3],kutular[4],kutular[5],kutular[6],kutular[7],kutular[8]],
    [kutular[9], kutular[10], kutular[11], kutular[12],kutular[13],kutular[14],kutular[15],kutular[16],kutular[17]],
    [kutular[18], kutular[19], kutular[20], kutular[21],kutular[22],kutular[23],kutular[24],kutular[25],kutular[26]],
    [kutular[27], kutular[28], kutular[29], kutular[30],kutular[31],kutular[32],kutular[33],kutular[34],kutular[35]],
    [kutular[36], kutular[37], kutular[38], kutular[39],kutular[40],kutular[41],kutular[42],kutular[43],kutular[44]],
    [kutular[45], kutular[46], kutular[47], kutular[48],kutular[49],kutular[50],kutular[51],kutular[52],kutular[53]],
    [kutular[54], kutular[55], kutular[56], kutular[57],kutular[58],kutular[59],kutular[60],kutular[61],kutular[62]],
    [kutular[63], kutular[64], kutular[65], kutular[66],kutular[67],kutular[68],kutular[69],kutular[70],kutular[71]],
    [kutular[72], kutular[73], kutular[74], kutular[75],kutular[76],kutular[77],kutular[78],kutular[79],kutular[80]]])


    sudokuNewSolved = cv.resize(sudokuNewSolved,(imgW,imgH))
    # for box in allBoxes:
    #     text = box[0]
    #     x = int(box[1])
    #     y = int(box[2])
    #     w = int(box[3])
    #     h = int(box [4])

    matrix2 = cv.getPerspectiveTransform(pointsAfter,pointsBefore)
    sudokuFinal = cv.warpPerspective(sudokuNewSolved,matrix2,(imgW,imgH))
    sudokuFinal = cv.addWeighted(img,0.5,sudokuFinal,10,2)

# GÖSTERME KISMI
imgList = stackImages(0.6,[[img, imgCanny, imgContoured,sudokuFinal],[sudoku,sudokuNew,sudokuNewSolved,imgBlank]])
cv.imshow("Pencere", imgList)
cv.waitKey(0)





