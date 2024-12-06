import cv2
import numpy as np



video_path = 'C:\\Users\\orhun\\Desktop\\GradProject\\autonomous_vehicle_yolov8\\Serit_Tespit\\Video\\road.mp4'

cap = cv2.VideoCapture(video_path)

top_left = (229, 399) # ust sol kose kordinat ust y kordinatlari esit
bottom_left = (122, 464) # alt sol kose kordinat alt y kordinatlari esit

top_right = (400,399) # ust sag kose kordinat
bottom_right = (480,464) # alt sag kose kordinat

point1 = np.float32([top_left, bottom_left, top_right, bottom_right]) # kus bakisi alinacak goruntunun kordinatlari roide sectigimiz noktalar
point2 = np.float32([[0,0], [0,480], [640,0], [640,480]]) # kus bakisi

color = (255,0,0)


while cap.isOpened():

    ret, frame = cap.read()
    frame = cv2.resize(frame, (640,480)) # kus bakisi icin imutils kullanmadik
    frame_copy = frame.copy()
    

    # -1 degeri cemberin ici dolu olur 1 degeri ise cemberin ici bos olur circle degeri son parametre
    #roi_selector ile belirledigimiz kordinatlarin uzerine cember cizdik

    cv2.circle(frame_copy, top_left, 5, color, -1) 
    cv2.circle(frame_copy, bottom_left, 5, color, -1)
    cv2.circle(frame_copy, top_right, 5, color, -1)
    cv2.circle(frame_copy, bottom_right, 5, color, -1)




    # kus bakisi almak icin perspektif donusumu yapmamiz gerekiyor
    matrix = cv2.getPerspectiveTransform(point1, point2) # perspektif donusumu icin matrix olusturduk
    transformed_birds_eye = cv2.warpPerspective(frame, matrix, (640,480)) # perspektif donusumu yaptik

    line_img = transformed_birds_eye.copy() # perspektif donusumu yapilan goruntuyu kopyaladik
   
    birds_eye_hsv  = cv2.cvtColor(transformed_birds_eye, cv2.COLOR_BGR2HSV) # perspektif donusumu yapilan goruntuyu hsv formatina cevirdik

    # Maskeleme icin renk araligi 
    lower_bound = np.array([0, 0, 200])
    upper_bound = np.array([255, 50, 255])

    # maske olusturma hsv uzerinde pikselleri yakalar
    birds_eye_mask = cv2.inRange(birds_eye_hsv, lower_bound, upper_bound)

    # maskeleme orjinal resimde mask da belirlenen piksellerle esler
    

    histogram = np.sum(birds_eye_mask[birds_eye_mask.shape[0]//2:, :], axis=0) # histogram olusturduk
    middle_point = np.int32(histogram.shape[0]/2) # histogramin orta noktasini belirledik
    left_side = np.argmax(histogram[:middle_point]) # histogramin sol tarafini belirledik
    right_side = np.argmax(histogram[middle_point:]) + middle_point  # histogramin sag tarafini belirledik

    # sliding window  (kayan kare)

    left_x = []
    right_x = []

    birds_eye_mask_copy = birds_eye_mask.copy() # maskeleme yapilan resmi kopyaladik

    starting_y = 480 # baslangic y degeri

    while starting_y > 0:

        img = birds_eye_mask[starting_y-40:starting_y, left_side-50:left_side+50] # resmin sol tarafini taradik   
        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                center_x = np.int32(M["m10"]/M["m00"])
                center_y = np.int32(M["m01"]/M["m00"])
                left_side = left_side - 50 + center_x

        img = birds_eye_mask[starting_y-40:starting_y, right_side-50:right_side+50] # resmin sag tarafini taradik  
        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                center_x = np.int32(M["m10"]/M["m00"])
                center_y = np.int32(M["m01"]/M["m00"])
                right_side = right_side - 50 + center_x
        
        cv2.rectangle(birds_eye_mask_copy, (left_side-60, starting_y),
                       (left_side+60, starting_y-40), (255,255,255), 2)
        
        cv2.rectangle(birds_eye_mask_copy, (right_side-60, starting_y), 
                      (right_side+60, starting_y-40), (255,255,255), 2)

        starting_y = starting_y - 40


    gray = cv2.cvtColor(line_img, cv2.COLOR_BGR2GRAY) # gri tonlama yaptik

    blur = cv2.GaussianBlur(gray, (5,5), 0) # gurultu azaltmak icin gaussian blur uyguladik

    edges = cv2.Canny(blur, 50, 150) # kenar tespiti icin canny edge detection uyguladik

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=150) # hough transform ile cizgileri bulduk

    birds_eye_mask_copy = cv2.cvtColor(birds_eye_mask_copy, cv2.COLOR_GRAY2BGR) # maskeleme yapilan resmi bgr formatina cevirdik
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0] # cizgileri cizdik
            cv2.line(line_img, (x1, y1), (x2, y2), (0,255,0),6)
            cv2.line(birds_eye_hsv, (x1, y1), (x2, y2), (0,0,0),6)
            cv2.line(birds_eye_mask_copy, (x1, y1), (x2, y2), (255,0,255),2)
            
        


    cv2.imshow("Frame", frame_copy)
    cv2.imshow("Birds Eye", transformed_birds_eye)
    cv2.imshow("HSV", birds_eye_hsv)
    cv2.imshow("Mask", birds_eye_mask)
    cv2.imshow("Sliding Window", birds_eye_mask_copy)
    cv2.imshow("Line Image", line_img)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()