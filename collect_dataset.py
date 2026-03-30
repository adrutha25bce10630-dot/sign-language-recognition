import cv2
import os

# Classes A-Z and 0-9
classes = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

dataset_path = "dataset"

# Create folders if they don't exist
for c in classes:
    path = os.path.join(dataset_path, c)
    os.makedirs(path, exist_ok=True)

# Choose class
current_class = input("Enter the sign to collect (A-Z or 0-9): ").upper()

save_path = os.path.join(dataset_path, current_class)

cap = cv2.VideoCapture(0)

count = len(os.listdir(save_path))

print("Press S to save image")
print("Press Q to quit")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Region of interest
    x1, y1, x2, y2 = 100, 100, 300, 300
    roi = frame[y1:y2, x1:x2]

    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

    cv2.putText(frame,
                f"Class: {current_class}  Count: {count}",
                (10,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2)

    cv2.imshow("Dataset Collector", frame)

    key = cv2.waitKey(1)

    # Save image
    if key == ord('s'):
        img = cv2.resize(roi,(64,64))
        filename = os.path.join(save_path, f"{count}.jpg")
        cv2.imwrite(filename, img)

        count += 1
        print("Saved:", filename)

    # Quit
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
