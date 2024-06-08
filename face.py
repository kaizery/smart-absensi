import cv2, os, numpy as np
import tkinter as tk
from PIL import ImageTk, Image
from datetime import datetime

def selesai1():
    intructions.config(text="Rekam Data Telah Selesai!")

def selesai2():
    intructions.config(text="Training Wajah Telah Selesai!")

def selesai3():
    intructions.config(text="Absensi Telah Dilakukan")

def generateNewID(metadata_file):
    # Membaca file metadata.csv untuk mendapatkan ID yang telah digunakan
    existing_ids = set()
    if os.path.exists(metadata_file):
        with open(metadata_file, "r", encoding='utf-8-sig') as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) == 3:
                    existing_ids.add(int(parts[0]))
    
    # Mencari ID baru yang belum digunakan
    new_id = 1
    while new_id in existing_ids:
        new_id += 1
    
    return new_id

def rekamDataWajah():
    wajahDir = 'datawajah'
    if not os.path.exists(wajahDir):
        os.makedirs(wajahDir)
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)
    cam.set(4, 480)
    faceDetector = cv2.CascadeClassifier('face.xml')
    eyeDetector = cv2.CascadeClassifier('haarcascade_eye.xml')
    nama = entry1.get()
    kelas = entry3.get()
    
    # Generate new ID
    new_id = generateNewID("datawajah/metadata.csv")
    
    ambilData = 1

    current_time  = datetime.now().strftime("%Y%m%d%H%M%S")
    with open("datawajah/metadata.csv", "a") as f:
        f.write(f"{new_id},{nama},{kelas}\n")
        
    while True:
        retV, frame = cam.read()
        abuabu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceDetector.detectMultiScale(abuabu, 1.3, 5)
        for (x, y, w, h) in faces:
            # Crop dan simpan wajah yang terdeteksi
            namaFile = f"{new_id}_{nama}_{kelas}_{ambilData}.jpg"
            filepath = os.path.join(wajahDir, namaFile)
            print(f"saving image to file: {filepath}")
            cv2.imwrite(filepath, frame[y:y + h, x:x + w])  # Simpan area wajah yang dideteksi
            ambilData += 1
            
            # Gambarkan persegi panjang di sekitar wajah
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            
            roiabuabu = abuabu[y:y + h, x:x + w]
            roiwarna = frame[y:y + h, x:x + w]
            eyes = eyeDetector.detectMultiScale(roiabuabu)
            for (xe, ye, we, he) in eyes:
                cv2.rectangle(roiwarna, (xe, ye), (xe + we, ye + he), (0, 255, 255), 1)
                
        cv2.imshow('webcamku', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # jika menekan tombol q akan berhenti
            break
        elif ambilData > 50:  # Batasi jumlah gambar yang diambil
            break
    selesai1()
    cam.release()
    cv2.destroyAllWindows()  # untuk menghapus data yang sudah dibaca


def trainingWajah():
    wajahDir = 'datawajah'
    latihDir = 'latihwajah'
    metadataFile = os.path.join(wajahDir, 'metadata.csv')

    # Membaca metadata dari file CSV
    metadata = {}
    try:
        with open(metadataFile, 'r', encoding='utf-8-sig') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) == 3:
                    id, nama, kelas = parts
                    metadata[int(id)] = (nama, kelas)  # Gunakan ID sebagai kunci dan nama serta kelas sebagai nilai
    except FileNotFoundError:
        print("Error: File metadata.csv tidak ditemukan.")
        return
    except Exception as e:
        print(f"Error saat membaca file metadata: {e}")
        return

    # Fungsi untuk mengambil gambar dan label
    def getImageLabel(path, id):
        faceSamples = []
        faceIDs = []

        try:
            # Mencari semua gambar yang terkait dengan ID tertentu
            imagePaths = [os.path.join(path, file) for file in os.listdir(path) if file.startswith(f"{id}_")]
            for imagePath in imagePaths:
                PIL_img = Image.open(imagePath).convert('L')
                img_numpy = np.array(PIL_img, 'uint8')
                faceSamples.append(img_numpy)
                faceIDs.append(id)
        except Exception as e:
            print(f"Error saat membaca gambar: {e}")

        return faceSamples, faceIDs

    # Mengumpulkan semua sampel wajah dan label untuk setiap ID
    all_faceSamples = []
    all_faceIDs = []
    for id in metadata:
        samples, ids = getImageLabel(wajahDir, id)
        all_faceSamples.extend(samples)
        all_faceIDs.extend(ids)

    if len(all_faceSamples) == 0:
        print("Tidak ada data wajah untuk dilatih.")
        return

    # Latih pengenalan wajah
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(all_faceSamples, np.array(all_faceIDs))
    if not os.path.exists(latihDir):
        os.makedirs(latihDir)
    recognizer.save(os.path.join(latihDir, 'training.xml'))
    selesai2()


def markAttendance(name):
    with open("Absensi_Attendance.csv", 'r+') as f:
        namesDatalist = f.readlines()
        namelist = []
        for line in namesDatalist:
            entry = line.split(',')
            namelist.append(entry[0])
        if name not in namelist:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{entry2.get()},{entry3.get()},{dtString}')

def absensiWajah():
    wajahDir = 'datawajah'
    latihDir = 'latihwajah'
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)
    cam.set(4, 480)
    faceDetector = cv2.CascadeClassifier('face.xml')
    faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
    faceRecognizer.read(os.path.join(latihDir, 'training.xml'))
    font = cv2.FONT_HERSHEY_SIMPLEX
    data = {}
    
    # Load metadata
    with open("datawajah/metadata.csv", "r", encoding='utf-8-sig') as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) == 3:
                id, nama, kelas = parts
                data[int(id)] = (nama, kelas)
    
    minWidth = 0.1 * cam.get(3)
    minHeight = 0.1 * cam.get(4)

    while True:
        retV, frame = cam.read()
        frame = cv2.flip(frame, 1)
        abuabu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceDetector.detectMultiScale(abuabu, 1.2, 5, minSize=(round(minWidth), round(minHeight)))
        for (x, y, w, h) in faces:
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            faceID, confidence = faceRecognizer.predict(abuabu[y:y+h, x:x+w])
            confidence_percentage = round(100 - confidence)

            print(f"Detected ID: {faceID} with confidence: {confidence_percentage}%")  # Debug log

            if confidence_percentage > 10:  # Ambang batas pengenalan, misalnya 10%
                if faceID in data:
                    nama, kelas = data[faceID]
                    confidence_text = f"{confidence_percentage}%"
                    cv2.putText(frame, nama, (x + 5, y - 5), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(frame, confidence_text, (x + 5, y + h + 25), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
                    markAttendance(nama)  # Memanggil fungsi markAttendance dengan parameter nama saja
                else:
                    nama = "Tidak Diketahui"
                    kelas = "Tidak Diketahui"
                    confidence_text = f"{confidence_percentage}%"
                    cv2.putText(frame, nama, (x + 5, y - 5), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(frame, confidence_text, (x + 5, y + h + 25), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
                    markAttendance(nama)  # Memanggil fungsi markAttendance dengan parameter nama saja
            else:
                nama = "Tidak Diketahui"
                kelas = "Tidak Diketahui"
                confidence_text = f"{confidence_percentage}%"
                cv2.putText(frame, nama, (x + 5, y - 5), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, confidence_text, (x + 5, y + h + 25), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
        
        cv2.imshow('ABSENSI WAJAH', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # jika menekan tombol q akan berhenti
            break

    selesai3()
    cam.release()
    cv2.destroyAllWindows()


# GUI
root = tk.Tk()
# mengatur canvas (window tkinter)
canvas = tk.Canvas(root, width=700, height=400)
canvas.grid(columnspan=3, rowspan=8)
canvas.configure(bg="black")
# judul
judul = tk.Label(root, text="Face Attendance - Smart Absensi", font=("Roboto",34),bg="#242526", fg="white")
canvas.create_window(350, 80, window=judul)
# credit
made = tk.Label(root, text="Made by Alvin Aprianto", font=("Times New Roman",13), bg="black",fg="white")
canvas.create_window(360, 20, window=made)
# for entry data nama
entry1 = tk.Entry(root, font="Roboto")
canvas.create_window(457, 170, height=25, width=411, window=entry1)
label1 = tk.Label(root, text="Nama Siswa", font="Roboto", fg="white", bg="black")
canvas.create_window(90, 170, window=label1)
# for entry data nim
entry2 = tk.Entry(root, font="Roboto")
canvas.create_window(457, 210, height=25, width=411, window=entry2)
label2 = tk.Label(root, text="NIM", font="Roboto", fg="white", bg="black")
canvas.create_window(60, 210, window=label2)
# for entry data kelas
entry3 = tk.Entry(root, font="Roboto")
canvas.create_window(457, 250, height=25, width=411, window=entry3)
label3 = tk.Label(root, text="Kelas", font="Roboto", fg="white", bg="black")
canvas.create_window(65, 250, window=label3)

global intructions

# tombol untuk rekam data wajah
intructions = tk.Label(root, text="Welcome", font=("Roboto",15), fg="white", bg="black")
canvas.create_window(370, 300, window=intructions)
Rekam_text = tk.StringVar()
Rekam_btn = tk.Button(root, textvariable=Rekam_text, font="Roboto", bg="#20bebe", fg="white", height=1, width=15, command=rekamDataWajah)
Rekam_text.set("Take Images")
Rekam_btn.grid(column=0, row=7)

# tombol untuk training wajah
Rekam_text1 = tk.StringVar()
Rekam_btn1 = tk.Button(root, textvariable=Rekam_text1, font="Roboto", bg="#20bebe", fg="white", height=1, width=15, command=trainingWajah)
Rekam_text1.set("Training")
Rekam_btn1.grid(column=1, row=7)

# tombol absensi dengan wajah
Rekam_text2 = tk.StringVar()
Rekam_btn2 = tk.Button(root, textvariable=Rekam_text2, font="Roboto", bg="#20bebe", fg="white", height=1, width=20, command=absensiWajah)
Rekam_text2.set("Automatic Attendance")
Rekam_btn2.grid(column=2, row=7)

root.mainloop()
