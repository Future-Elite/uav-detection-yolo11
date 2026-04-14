from ultralytics import YOLO
import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":

    dataset = "merged"
    model_addr = "refined-models/yolo11-ablation4-CSPPC-ECA-SPPELAN.yaml"

    model = YOLO(model_addr).load("./runs/detect/merged/refined/best.pt")
    model.train(cfg=f'./configs/{dataset}-config.yaml')
