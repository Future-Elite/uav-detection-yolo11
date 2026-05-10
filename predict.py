from ultralytics import YOLO


if __name__ == '__main__':

    model = YOLO(r"./runs/detect/ablation-5/weights/best.pt")

    # model = YOLO("refined-models/yolo11-CSPPC-ECA-SPPELAN.yaml")
    # model.train(cfg='./configs/merged-config.yaml')

    results = model.val(data='../datasets/Airborne/data.yaml', split='test',
                        imgsz=640,
                        batch=1,
                        conf=0.001,
                        iou=0.5,
                        nms=True,
                        )
