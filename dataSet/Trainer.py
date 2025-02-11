from ultralytics import YOLO

def train_model():
    # Model yükleme
    model = YOLO("yolo11n.pt")
    
    # Eğitim
    model.train(
        data=r"C:\Users\orhun\Desktop\GradProject\autonomous_vehicle_yolov8\dataSet\data.yaml", 
        epochs=200, 
        imgsz=640, 
        workers=4, 
        batch=16, 
        device=0, 
        name="Trainers"
    )

if __name__ == "__main__":
    train_model()