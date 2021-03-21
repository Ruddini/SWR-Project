import cv2
import numpy as np

class LineDetector():
    """Klasa odpowiedzialna za analize obrazu, wyszukiwanie koloru (z 3 wybranuch kolorów linni),
        którego jest na danym obrazie najwięcej i wyznaczanie środka 'ciężkości' tego koloru na obrazie"""
    def __init__(self,vc = cv2.VideoCapture(0)):
        self.video_capture = vc
        self.ret , self.frame = self.video_capture.read()
        self.orginal_frame = self.frame
        self.color = 0
                #ta wartosc będzie 0 jeżeli nie rozpoznana będzie żadna linia
                # to będzie potrzebne do ustalenia prędkości robota
                # 1 kolor linni zielony
                # 2 kolor linni niebieski
                # 3 kolor linni czerwony
        self.green_BGR = [50, 200, 16]
        self.blue_BGR = [190,70,20]
        self.red_BGR = [60, 10, 200]
        self.tresh = 70


    def find_color(self,color,tresh,pic):
        col = np.array(color)
        #wyznaczanie odchyłek danego koloru
        self.min_BGR = col - tresh
        self.max_BGR = col + tresh
        mask_BGR = cv2.inRange(pic,self.min_BGR,self.max_BGR)
        self.pic_BGR = cv2.bitwise_and(pic, pic, mask=mask_BGR) # nałożenie obrazów tak aby wyróżniony kontur był zielony (opcjonalna funkcja z której koniec końcó nie korzystamy)
        return mask_BGR

    def find_contours(self,mask_BGR, color_lines):
        self.countours ,self.hierarchy = cv2.findContours(mask_BGR.copy(),1,cv2.CHAIN_APPROX_NONE)
        #warunek istnienia konturu danego koloru
        if len(self.countours)>0:
            self.max_cont = max(self.countours ,key=cv2.contourArea)
            #zapewnienie warunku dzielenia przez 0
            if cv2.moments(self.max_cont)['m00']>0:
                #wyznaczeie srodka ciężkośći znalezionego konturu
                self.center_x = int(cv2.moments(self.max_cont)['m10']/cv2.moments(self.max_cont)['m00'])
                self.center_y = int(cv2.moments(self.max_cont)['m01']/cv2.moments(self.max_cont)['m00'])
                cv2.line(self.frame, (self.center_x, 0), (self.center_x, 720), (255, 0, 0), 1)
                cv2.line(self.frame, (0, self.center_y), (1280, self.center_y), (255, 0, 0), 1)
            #narysowanie konturów
            cv2.drawContours(self.frame, self.countours, -1, color_lines, 1)
            return self.frame

    def detection_color_line(self):
        pix_num = []
        # tworzenie maski dla poszczególnych kolorów
        green_col = self.find_color(self.green_BGR, 70, self.frame)
        pix_num.append(cv2.countNonZero(green_col))
        blue_col = self.find_color(self.blue_BGR, 70, self.frame)
        pix_num.append(cv2.countNonZero(blue_col))
        red_col = self.find_color(self.red_BGR, 70, self.frame)
        pix_num.append(cv2.countNonZero(red_col))

        # warunek że żadnego koloru nie jest więcej niż innego
        if len(set(pix_num))==3:
            max_white_pix = pix_num.index(max(pix_num))  # powinieniem dać warunek że wartość musi być większa od jakiejś wartości
            del pix_num
            # wybór koloru, którego jest najwięcej i ustawienie flagi color na odpwoednią wartość
            if max_white_pix == 0:
                self.color = 1
                pic = self.find_contours(green_col,self.green_BGR)
            elif max_white_pix == 1:
                self.color = 2
                pic = self.find_contours(blue_col,self.blue_BGR)
            elif max_white_pix == 2:
                self.color = 3
                pic = self.find_contours(red_col,self.red_BGR)
        else:
            pic = self.frame
        return pic





