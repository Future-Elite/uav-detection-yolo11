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
    model = YOLO("./runs/detect/merged/refined/best.pt")  # 0.97
    # model = YOLO("./runs/detect/merged/refined/best-enhanced.pt")
    # model.train(cfg='./configs/merged-config.yaml')

    # 遍历test文件夹，对每张图片进行预测并得出map50
    # test_images_dir = f"../datasets/{project}/test/images"
    # output_dir = f"./predictions_{project}"
    # os.makedirs(output_dir, exist_ok=True)
    # for img_name in os.listdir(test_images_dir):
    #     if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
    #         img_path = os.path.join(test_images_dir, img_name)
    #         results = model.predict(source=img_path, save=False, save_txt=False,
    #         save_conf=True, augment=True, agnostic_nms=True)
    #         for result in results:
    #             result.save(os.path.join(output_dir, img_name))
    #
    # print(f"Predictions saved to {output_dir}")

    results = model.val(data='../datasets/Airborne/data.yaml', split='test',
                        imgsz=640,
                        batch=16,
                        conf=0.001,
                        iou=0.5,
                        )
