from roboflow import Roboflow
import cv2

rf = Roboflow(api_key="rf_47Hw8nCN3YSt0ImLFhp5mJAFyNC3")
project = rf.workspace().project("ia-frutas-ic-keccy")  
model = project.version(1).model                    

image_path = "frutas.jpg"

prediction = model.predict(image_path, confidence=40, overlap=30).json()

image = cv2.imread(image_path)

for det in prediction["predictions"]:
    x, y = int(det["x"]), int(det["y"])
    w, h = int(det["width"]), int(det["height"])
    label = det["class"]
    confidence = det["confidence"]

    x1 = x - w // 2
    y1 = y - h // 2
    x2 = x + w // 2
    y2 = y + h // 2

    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        image,
        f"{label} ({confidence:.2f})",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

cv2.imshow("Detecção de Frutas", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
