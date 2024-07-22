import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np

def stackImages(imgArray, scale):
     # Görüntü dizisindeki satır ve sütun sayısı
    rows = len(imgArray)
    cols = len(imgArray[0]) if isinstance(imgArray[0], list) else 1
    # Görüntü genişliği ve yüksekliği
    width, height = imgArray[0][0].shape[1], imgArray[0][0].shape[0]
    resized_images = []

    for row in imgArray:
        resized_row = []
        for img in (row if isinstance(row, list) else [row]):
            # Görüntüyü ölçek temelinde yeniden boyutlandır
            resized_img = cv2.resize(img, (0, 0), None, scale, scale)
            # Gri tonlamalı görüntüyü BGR'ye dönüştürme gerekirse
            resized_img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2BGR) if len(resized_img.shape) == 2 else resized_img
            resized_row.append(resized_img)
        # Görüntüleri yatay olarak birleştir
        resized_images.append(np.hstack(resized_row))

    return np.vstack(resized_images)

def reorder(points):
    # Noktaları yeniden şekillendir
    points = points.reshape((4, 2))
    # Toplamları ve farkları hesapla
    add = points.sum(1)
    diff = np.diff(points, axis=1)
     # Noktaları belirli bir sırada döndür
    return np.array([points[np.argmin(add)], points[np.argmin(diff)], points[np.argmax(diff)], points[np.argmax(add)]])

def rectContour(contours):
    # Dörtgen şeklindeki konturları seç ve alanlarına göre sırala
    return sorted([cnt for cnt in contours if cv2.contourArea(cnt) > 50 and len(cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)) == 4], key=cv2.contourArea, reverse=True)

def splitBoxes(img, numRows, numCols):
    # Görüntüyü bölerek kutuları elde et
    return [box for r in np.vsplit(img, numRows) for box in np.hsplit(r, numCols)]

def drawGrid(img, numRows, numCols):
    # Hücre genişliği ve yüksekliği
    secW, secH = img.shape[1] // numCols, img.shape[0] // numRows
     # Yatay ve dikey çizgileri çiz
    for i in range(numRows + 1):
        cv2.line(img, (0, secH * i), (img.shape[1], secH * i), (255, 255, 0), 2)
        cv2.line(img, (secW * i, 0), (secW * i, img.shape[0]), (255, 255, 0), 2)
    return img

def showAnswers(img, myIndex, grading, ans, numRows, numCols):
     # Hücre genişliği ve yüksekliği
    secW, secH = img.shape[1] // numCols, img.shape[0] // numRows
    for x in range(numRows):
        # Doğru ve yanlış yanıtlar için merkez noktalar ve renkler
        cX, cY = (myIndex[x] * secW) + secW // 2, (x * secH) + secH // 2
        color = (0, 255, 0) if grading[x] else (0, 0, 255)
        cv2.circle(img, (cX, cY), 50, color, cv2.FILLED)
        if not grading[x]:
            cv2.circle(img, ((ans[x] * secW) + secW // 2, cY), 20, (0, 255, 0), cv2.FILLED)

def select_image(is_answer_key):
    global answer_key_path, student_path
    # Dosya seçme penceresini aç
    path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
    if path:
        if is_answer_key:
            answer_key_path = path
            img = Image.open(answer_key_path).resize((400, 400))
        else:
            student_path = path
            img = Image.open(student_path).resize((400, 400))
        
        img = ImageTk.PhotoImage(img)
        if is_answer_key:
            panel_answer.config(image=img)
            panel_answer.image = img
        else:
            panel_student.config(image=img)
            panel_student.image = img

def process_image():
    global answer_key_path, student_path, imgFinal, score, correct, incorrect
    if answer_key_path and student_path:
        # Görüntü boyutları ve soru sayısı
        heightImg, widthImg = 700, 700
        questions, choices = 10, 5

        img_answer = cv2.imread(answer_key_path)
        img_student = cv2.imread(student_path)
        if img_answer is None or img_student is None:
            print("Resim dosyası yüklenemedi.")
            return

        img_answer = cv2.resize(img_answer, (widthImg, heightImg))
        img_student = cv2.resize(img_student, (widthImg, heightImg))

        imgFinal = img_student.copy()
        imgGray_answer = cv2.cvtColor(img_answer, cv2.COLOR_BGR2GRAY)
        imgGray_student = cv2.cvtColor(img_student, cv2.COLOR_BGR2GRAY)

        imgCanny_answer = cv2.Canny(cv2.GaussianBlur(imgGray_answer, (5, 5), 1), 10, 70)
        imgCanny_student = cv2.Canny(cv2.GaussianBlur(imgGray_student, (5, 5), 1), 10, 70)

        contours_answer, _ = cv2.findContours(imgCanny_answer, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours_student, _ = cv2.findContours(imgCanny_student, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        rectCon_answer = rectContour(contours_answer)
        rectCon_student = rectContour(contours_student)

        if len(rectCon_answer) < 1 or len(rectCon_student) < 1:
            print("Yeterli kontur bulunamadı.")
            return

        biggestPoints_answer = reorder(cv2.approxPolyDP(rectCon_answer[0], 0.02 * cv2.arcLength(rectCon_answer[0], True), True))
        biggestPoints_student = reorder(cv2.approxPolyDP(rectCon_student[0], 0.02 * cv2.arcLength(rectCon_student[0], True), True))

        imgWarpColored_answer = cv2.warpPerspective(img_answer, cv2.getPerspectiveTransform(np.float32(biggestPoints_answer), np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])), (widthImg, heightImg))
        imgWarpColored_student = cv2.warpPerspective(img_student, cv2.getPerspectiveTransform(np.float32(biggestPoints_student), np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])), (widthImg, heightImg))

        imgThresh_answer = cv2.threshold(cv2.cvtColor(imgWarpColored_answer, cv2.COLOR_BGR2GRAY), 150, 255, cv2.THRESH_BINARY_INV)[1]
        imgThresh_student = cv2.threshold(cv2.cvtColor(imgWarpColored_student, cv2.COLOR_BGR2GRAY), 150, 255, cv2.THRESH_BINARY_INV)[1]

        myPixelVal_answer = np.array([cv2.countNonZero(box) for box in splitBoxes(imgThresh_answer, questions, choices)]).reshape(questions, choices)
        myPixelVal_student = np.array([cv2.countNonZero(box) for box in splitBoxes(imgThresh_student, questions, choices)]).reshape(questions, choices)

        myIndex_answer = [np.argmax(myPixelVal_answer[x]) for x in range(questions)]
        myIndex_student = [np.argmax(myPixelVal_student[x]) for x in range(questions)]

        grading = [1 if myIndex_answer[x] == myIndex_student[x] else 0 for x in range(questions)]
        score = (sum(grading) / questions) * 100
        correct = sum(grading)
        incorrect = questions - correct

        showAnswers(imgWarpColored_student, myIndex_student, grading, myIndex_answer, questions, choices)
        imgRawDrawings = np.zeros_like(imgWarpColored_student)
        showAnswers(imgRawDrawings, myIndex_student, grading, myIndex_answer, questions, choices)
        imgInvWarp = cv2.warpPerspective(imgRawDrawings, cv2.getPerspectiveTransform(np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]]), np.float32(biggestPoints_student)), (widthImg, heightImg))

        imgRawGrade = np.zeros((150, 325, 3), np.uint8)
        cv2.putText(imgRawGrade, str(int(score)) + "%", (70, 100), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 0, 0), 3)
        imgInvGradeDisplay = cv2.warpPerspective(imgRawGrade, cv2.getPerspectiveTransform(np.float32([[0, 0], [325, 0], [0, 150], [325, 150]]), np.float32(biggestPoints_student)), (widthImg, heightImg))

        imgFinal = cv2.addWeighted(imgFinal, 1, imgInvWarp, 1, 0)
        imgFinal = cv2.addWeighted(imgFinal, 1, imgInvGradeDisplay, 1, 0)



        # Sonuçları görüntüleme
        display_result()

def display_result():
    global imgFinal, score, correct, incorrect
    result_window = tk.Toplevel()
    result_window.title("Sonuç")
    result_window.geometry("700x900")

    # Siyah arka plan rengi ve beyaz metin rengi için stilleri ayarla
    result_window.configure(bg="#000000")
    ttk.Style().configure("TLabel", background="#000000", foreground="#ffffff", font=("Helvetica", 14))

    # Score değerini ondalık kısmı olmadan yazdır
    score_int = int(score)  # Score'u tam sayıya dönüştür
    result_label = ttk.Label(result_window, text=f"Skor: {score_int}\nDoğru: {correct}\nYanlış: {incorrect}")
    result_label.pack(pady=5)

    result_img = cv2.cvtColor(imgFinal, cv2.COLOR_BGR2RGB)  
    result_img = Image.fromarray(result_img)  
    result_img = ImageTk.PhotoImage(result_img)  
    result_panel = ttk.Label(result_window, image=result_img, background="#000000")
    result_panel.image = result_img  
    result_panel.pack(pady=10)

# Dikey beyaz çizgi çizme fonksiyonu
def draw_vertical_line(img):
    height, width, _ = img.shape
    cv2.line(img, (width//2, 0), (width//2, height), (255, 255, 255), 2)
    return img

# Arka planı siyah yapma fonksiyonu
def black_bg_image(img):
    return np.zeros_like(img)

# Ana pencere oluşturma ve arayüz öğelerini yerleştirme
root = tk.Tk()
root.title("Cevap Anahtarı Karşılaştırma Uygulaması")
root.geometry("900x600")  # Yüksekliği artırıldı

style = ttk.Style()
style.configure("TButton", font=("Helvetica", 10), padding=10)
style.configure("TLabel", font=("Helvetica", 12), background="#f0f0f0", foreground="#000000")  # Arka plan siyah, yazılar beyaz

# Arka plan görüntüsü
bg_image = black_bg_image(np.zeros((600, 900, 3), dtype=np.uint8))  # Boyutlar değiştirildi
bg_image = draw_vertical_line(bg_image)
bg_image = Image.fromarray(bg_image)
bg_image = ImageTk.PhotoImage(bg_image)
background_label = ttk.Label(root, image=bg_image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

# Cevap anahtarı ve öğrenci cevapları için panel oluşturma
panel_answer = ttk.Label(root)
panel_answer.place(x=20, y=50)  # Yer değiştirildi

panel_student = ttk.Label(root)
panel_student.place(x=480, y=50)  # Yer değiştirildi

# Butonlar oluşturma
select_button_answer = ttk.Button(root, text="Cevap Anahtarı Seç", command=lambda: select_image(True))
select_button_answer.place(x=130, y=550)  # Yer değiştirildi

select_button_student = ttk.Button(root, text="Öğrenci Cevapları Seç", command=lambda: select_image(False))
select_button_student.place(x=630, y=550)  # Yer değiştirildi

process_button = ttk.Button(root, text="İşle", command=process_image)
process_button.place(x=400, y=550)  # Yer değiştirildi

root.mainloop()