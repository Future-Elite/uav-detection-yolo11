from ultralytics import YOLO


if __name__ == '__main__':

    model = YOLO("./runs/detect/merged/refined-enhanced3/weights/best.pt")  # 0.929

    model.export(format="engine", imgsz=640, batch=16, conf=0.001, iou=0.5, int8=True,data='../datasets/Airborne/data.yaml', split='test')

