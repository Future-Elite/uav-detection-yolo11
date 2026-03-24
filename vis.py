import os
import cv2

# ================= 配置区 =================
DATASET_ROOT = "../datasets/UAV2/train"   # 改成你的数据集路径
IMAGE_DIR = os.path.join(DATASET_ROOT, "images")
LABEL_DIR = os.path.join(DATASET_ROOT, "labels")
OUTPUT_DIR = os.path.join(DATASET_ROOT, "vis")

IMG_EXTS = [".jpg", ".png", ".jpeg"]
# =========================================


def yolo_to_xyxy(box, img_w, img_h):
    cls, x, y, w, h = box
    x *= img_w
    y *= img_h
    w *= img_w
    h *= img_h

    x1 = int(x - w / 2)
    y1 = int(y - h / 2)
    x2 = int(x + w / 2)
    y2 = int(y + h / 2)
    return int(cls), x1, y1, x2, y2


def visualize_one(img_path, label_path, save_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"[ERROR] 读取失败: {img_path}")
        return

    h, w = img.shape[:2]

    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            parts = list(map(float, parts))
            cls, x1, y1, x2, y2 = yolo_to_xyxy(parts, w, h)

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img,
                str(cls),
                (x1, max(y1 - 5, 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    images = [
        f for f in os.listdir(IMAGE_DIR)
        if os.path.splitext(f)[1].lower() in IMG_EXTS
    ]

    print(f"Found {len(images)} images")

    for img_name in images:
        img_path = os.path.join(IMAGE_DIR, img_name)
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(LABEL_DIR, label_name)
        save_path = os.path.join(OUTPUT_DIR, img_name)

        visualize_one(img_path, label_path, save_path)

    print("Visualization done.")


if __name__ == "__main__":
    main()
