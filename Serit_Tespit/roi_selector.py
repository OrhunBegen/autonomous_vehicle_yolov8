import numpy as np
import cv2 as cv

##############################
# 1. OpenCV -> read a frame: cap.read()
# 2. Mouse Event -> define mouse event: cv2.EVENT_LBUTTONDOWN
# 3. Show Reasults ->
##############################

color =(0,255,0)
font = cv.FONT_HERSHEY_SIMPLEX
clicked_point = []

path = 'video/road.mp4' # video yolu
cap = cv.VideoCapture(path) #videoyu okuma opencv ile

ret, img = cap.read()
img = cv.resize(img,(640,480))


def click_event(event ,x ,y ,flags ,param):
    if event == cv.EVENT_LBUTTONDOWN:
        #print('Kordinatlar : ({},{})'.format(x,y))
        cv.circle(img, (x,y), 5, color, -1)
        cv.putText(img,f'({x}, {y})',(x, y-10),font,0.5,color,2)

        cv.imshow('Test', img)
        clicked_point.append((x,y))

cv.imshow('Test', img)
cv.setMouseCallback('Test', click_event)


if cv.waitKey(0) == 27:
    cv.imwrite('coordinates.png',img)

    for point in clicked_point:
        print(f'Kordinat : {point}')
cv.destroyAllWindows()