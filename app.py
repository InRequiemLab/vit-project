from flask import Flask, render_template, Response
import cv2
import sys
sys.path.insert(0, './training') 
from detection import DETECT

model = DETECT("models/detr_model.pt")

app = Flask(__name__)

cap = cv2.VideoCapture(0)  
def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        results = model(frame)
        for result in results:
            if hasattr(result, 'names') and result.names:
                class_ids = result.boxes.cls.cpu().numpy()
                boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                confidences = result.boxes.conf.cpu().numpy()

                for i, class_id in enumerate(class_ids):
                    name = result.names.get(int(class_id), "Unknown")
                    box = boxes[i]
                    confidence = confidences[i]

                    if confidence >= 0.8:
                        x1, y1, x2, y2 = box
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(frame, f"{name} ({confidence:.2f})", (int(x1), int(y1) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
