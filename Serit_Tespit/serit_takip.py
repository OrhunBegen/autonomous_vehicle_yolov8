import cv2 
import numpy as np

top_left = (230,406)
bottom_left = (160,454)
top_right =(376,403)
bottom_right = (435,454)

pts1 = np.float32([top_left, bottom_left, top_right, bottom_right])
pts2 = np.float32([[0,0], [0,480], [640,0], [640,480]])

color = (0,255,0)

path = 'video/road.mp4' # Video Yolu
cap = cv2.VideoCapture(path) # Video Okuma

if not cap.isOpened():
    print('Error: Could not open video.')
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print('Error: Could not read frame.')
        break
    
    frame = cv2.resize(frame, (640, 480))

    # Kuş Bakışı Görünüm
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    transformed_frame = cv2.warpPerspective(frame, matrix, (640, 480))

    # Gri Tonlama
    gray = cv2.cvtColor(transformed_frame, cv2.COLOR_BGR2GRAY)

    # Gaussian Blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Kenar Tespiti
    edges = cv2.Canny(blur, 50, 150)

    # ROI (İlgi Bölgesi) Belirleme
    mask = np.zeros_like(edges)
    height, width = edges.shape
    polygon = np.array([[(0, height), (width, height), (width//2, height//2)]])
    cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Hough Transform ile Çizgileri Bulma
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 50, maxLineGap=50, minLineLength=50)

    # Çizgileri Çizme
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)

    # Orijinal Görüntü ile Çizgileri Birleştirme
    combo = cv2.addWeighted(transformed_frame, 0.8, line_image, 1, 0)

    # Sonuçları Gösterme
    cv2.imshow('Frame', frame)
    cv2.imshow('Transformed Frame', transformed_frame)
    cv2.imshow('Edges', edges)
    cv2.imshow('Masked Edges', masked_edges)
    cv2.imshow('Result', combo)
    
    if cv2.waitKey(0) == 27:
        break

cv2.destroyAllWindows()
