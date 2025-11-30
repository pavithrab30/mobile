from ultralytics import YOLO
import cv2

# Load model
model = YOLO("best.pt")

def detect_pothole_image(image_path):
    img = cv2.imread(image_path)

    if img is None:
        print("❌ ERROR: Could not load image:", image_path)
        return

    results = model(img, conf=0.25)

    # Force thicker bounding boxes
    annotated = results[0].plot(line_width=3)

    cv2.imshow("Pothole Detection", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_pothole_video(video_path):
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.25)
        annotated = results[0].plot(line_width=3)

        cv2.imshow("Pothole Detection", annotated)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("1. Detect pothole in image")
    print("2. Detect pothole in video")

    ch = input("Enter choice: ")

    if ch == "1":
        detect_pothole_image("image.png")
    else:
        detect_pothole_video("sample_video.mp4")
