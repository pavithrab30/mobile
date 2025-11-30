import cv2

# Load Haar Cascade for number plate
plate_cascade = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")

# -------------- IMAGE FUNCTION ---------------- #
def blur_number_plate_image(image_path):
    img = cv2.imread(image_path)

    if img is None:
        print("❌ Image not found:", image_path)
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect number plates
    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    print("🔍 Plates detected:", len(plates))

    # Blur detected plates
    for (x, y, w, h) in plates:
        plate_roi = img[y:y+h, x:x+w]
        blurred = cv2.GaussianBlur(plate_roi, (35, 35), 30)
        img[y:y+h, x:x+w] = blurred

    cv2.imshow("Blurred Number Plate - Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# -------------- VIDEO FUNCTION ---------------- #
def blur_number_plate_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("❌ Unable to open video:", video_path)
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        plates = plate_cascade.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in plates:
            region = frame[y:y+h, x:x+w]
            blur = cv2.GaussianBlur(region, (35, 35), 30)
            frame[y:y+h, x:x+w] = blur

        cv2.imshow("Blurred Number Plate - Video", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ------------------ MAIN --------------------- #
if __name__ == "__main__":
    print("\n=== Number Plate Blur Project ===")
    print("1. Blur number plate in IMAGE")
    print("2. Blur number plate in VIDEO")
    
    choice = input("Enter choice (1 or 2): ")

    if choice == "1":
        path = input("Enter image path: ")
        blur_number_plate_image(path)

    elif choice == "2":
        path = input("Enter video path: ")
        blur_number_plate_video(path)

    else:
        print("❌ Invalid choice. Exiting.")
