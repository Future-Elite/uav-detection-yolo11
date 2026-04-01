from ultralytics import YOLO


if __name__ == '__main__':
    # model = YOLO("./runs/detect/merged/refined-45re/weights/best.pt")
    # model = YOLO("./runs/detect/merged/refined-focaleiou-enhancedp3/weights/best.pt")  # 0.849
    # model = YOLO("./runs/detect/merged/yolo11s-ori/weights/best.pt")  # 0.844
    # model = YOLO("./runs/detect/merged/refined-siou-enhancedp3/weights/best.pt")  # 0.865
    # model = YOLO("./runs/detect/merged/refined-24/weights/best.pt")  # 0.856
    # model = YOLO("./runs/detect/merged/refined/yolo11s-CSPPC-ECA-SPPELAN-SIoU.pt")  # 0.865
    # model = YOLO("./runs/detect/merged/refined/yolo11s-SIoU-enhancedP3.pt")  # 0.875
    # model = YOLO("./runs/detect/merged/refined-yolo11n-CSPPC-ECA-SPPELAN/weights/best.pt")  # 0.866 同0.875
    # model = YOLO(r"C:\Users\dthqe\Desktop\refined\weights\best.pt")  # 0.857
    # model = YOLO("./runs/detect/merged/refined-mosaic/weights/best.pt")  # 0.854
    # model = YOLO(r"D:\Workspace\Thesis\Code\runs\detect\merged\refined-enhanced-origin\weights\best.pt")  # 0.837
    model = YOLO(r"D:\Workspace\Thesis\Code\runs\detect\merged\refined-enhanced3\weights\best.pt")  # 0.929

    # model.train(cfg='./configs/merged-config.yaml')


    results = model.val(data='../datasets/Airborne/data.yaml', split='test',
                        imgsz=640,
                        batch=16,
                        conf=0.001,
                        iou=0.5,
                        nms=True,
                        )
