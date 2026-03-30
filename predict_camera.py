import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("model.h5")

# Labels must match dataset folders
labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    # Define box for hand
    x1, y1, x2, y2 = 100,100,300,300
    roi = frame[y1:y2, x1:x2]

    # Draw rectangle
    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

    # Preprocess image
    img = cv2.resize(roi,(64,64))
    img = img.astype("float32")/255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img, verbose=0)
    #print(prediction)
    print(prediction.shape)
    index = np.argmax(prediction)
    confidence = float(prediction[0][index])

    threshold = 0.01   

    # Get predicted label
    if confidence > threshold and index < len(labels):
        label = f"{labels[index]} ({confidence:.2f})"
    else:
        label = "Unknown"
    # Show result
    cv2.putText(frame,str(label),(100,90),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,(0,255,0),3)

    cv2.imshow("Sign Language Recognition",frame)

    key = cv2.waitKey(1)

    if key == 27:   # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
