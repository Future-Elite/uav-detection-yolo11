"""
空中障碍物实时检测界面 - Flask后端
检测类别：plane(飞机)、bird(鸟类)、drone(无人机)、helicopter(直升机)
"""

import cv2
import torch
import time
import threading
import os
from flask import Flask, render_template, Response, request, jsonify
from ultralytics import YOLO
from collections import defaultdict

app = Flask(__name__)

# 全局变量
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'runs/detect/merged/refined/best.pt')

# 检测类别
CLASS_NAMES = {0: 'plane', 1: 'bird', 2: 'drone', 3: 'helicopter'}
CLASS_COLORS = {
    'plane': (255, 0, 0),       # 蓝色
    'bird': (0, 255, 0),        # 绿色
    'drone': (0, 0, 255),       # 红色
    'helicopter': (255, 255, 0)  # 青色
}

# 检测参数（线程安全）
params_lock = threading.Lock()
detection_params = {
    'conf': 0.25,
    'iou': 0.45,
    'classes': [0, 1, 2, 3],  # 所有类别
    'source_type': 'camera',  # 'camera' or 'video'
    'video_path': None,
    'is_running': False
}

# 统计信息
stats = {
    'fps': 0,
    'total_detections': 0,
    'class_counts': defaultdict(int),
    'frame_count': 0
}

# 视频源
video_source = None
video_lock = threading.Lock()
need_new_source = threading.Event()  # 标记需要重新获取视频源

# 加载模型
print("正在加载YOLO11模型...")
model = YOLO(MODEL_PATH)
print("模型加载完成！")


def create_video_capture(source):
    """创建视频捕获对象，Windows平台优先使用MSMF后端避免FFmpeg多线程问题"""
    if isinstance(source, str):
        # 视频文件：Windows优先使用MSMF后端
        if os.name == 'nt':
            cap = cv2.VideoCapture(source, cv2.CAP_MSMF)
            if not cap.isOpened():
                # MSMF失败则回退到默认后端
                cap = cv2.VideoCapture(source)
        else:
            cap = cv2.VideoCapture(source)
    else:
        # 摄像头索引
        if os.name == 'nt':
            cap = cv2.VideoCapture(source, cv2.CAP_MSMF)
            if not cap.isOpened():
                cap = cv2.VideoCapture(source)
        else:
            cap = cv2.VideoCapture(source)
    return cap


def release_video_source():
    """安全释放视频源"""
    global video_source
    with video_lock:
        if video_source is not None:
            try:
                video_source.release()
            except Exception as e:
                print(f"释放视频源时出错: {e}")
            video_source = None


def get_video_source():
    """获取当前视频源"""
    global video_source
    with video_lock:
        with params_lock:
            source_type = detection_params['source_type']
            video_path = detection_params['video_path']

        # 先释放旧的视频源
        if video_source is not None:
            try:
                video_source.release()
            except Exception:
                pass
            video_source = None

        # 创建新的视频源
        if source_type == 'video' and video_path and os.path.exists(video_path):
            video_source = create_video_capture(video_path)
        else:
            video_source = create_video_capture(0)  # 默认摄像头

        need_new_source.clear()
        return video_source


def generate_frames():
    """生成MJPEG帧流"""
    global video_source, stats

    prev_time = time.time()
    frame_count = 0

    while True:
        with params_lock:
            if not detection_params['is_running']:
                time.sleep(0.1)
                continue
            current_params = detection_params.copy()

        # 检查是否需要重新获取视频源
        if need_new_source.is_set() or video_source is None:
            get_video_source()

        # 线程安全地读取帧
        with video_lock:
            if video_source is None or not video_source.isOpened():
                print("无法打开视频源")
                time.sleep(1)
                continue
            success, frame = video_source.read()

        if not success:
            # 视频结束或读取失败
            if current_params['source_type'] == 'video':
                # 视频播放完毕，重新开始
                release_video_source()
                get_video_source()
                continue
            else:
                time.sleep(0.1)
                continue

        # 执行检测
        with params_lock:
            conf = detection_params['conf']
            iou = detection_params['iou']
            classes = detection_params['classes']

        try:
            results = model(frame, conf=conf, iou=iou, classes=classes, verbose=False)

            # 绘制检测结果
            annotated_frame = frame.copy()
            current_class_counts = defaultdict(int)

            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls_id = int(box.cls[0])
                        conf_score = float(box.conf[0])
                        cls_name = CLASS_NAMES.get(cls_id, 'unknown')
                        current_class_counts[cls_name] += 1

                        # 绘制边界框
                        color = CLASS_COLORS.get(cls_name, (0, 255, 0))
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                        # 绘制标签
                        label = f"{cls_name}: {conf_score:.2f}"
                        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10),
                                     (x1 + label_size[0], y1), color, -1)
                        cv2.putText(annotated_frame, label, (x1, y1 - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # 计算FPS
            frame_count += 1
            current_time = time.time()
            elapsed = current_time - prev_time

            if elapsed >= 1.0:
                with params_lock:
                    stats['fps'] = frame_count / elapsed
                    stats['total_detections'] = sum(current_class_counts.values())
                    stats['class_counts'] = dict(current_class_counts)
                    stats['frame_count'] += frame_count
                frame_count = 0
                prev_time = current_time

            # 在帧上显示FPS
            fps_text = f"FPS: {stats['fps']:.1f}"
            cv2.putText(annotated_frame, fps_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 显示检测数量
            det_text = f"Detections: {sum(current_class_counts.values())}"
            cv2.putText(annotated_frame, det_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # 编码为JPEG
            ret, buffer = cv2.imencode('.jpg', annotated_frame,
                                       [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        except Exception as e:
            print(f"检测错误: {e}")
            continue


@app.route('/')
def index():
    """主页面"""
    return render_template('index.html',
                          class_names=CLASS_NAMES,
                          class_colors=CLASS_COLORS)


@app.route('/video_feed')
def video_feed():
    """MJPEG视频流"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/update_params', methods=['POST'])
def update_params():
    """更新检测参数"""
    with params_lock:
        data = request.get_json()

        if 'conf' in data:
            detection_params['conf'] = float(data['conf'])

        if 'iou' in data:
            detection_params['iou'] = float(data['iou'])

        if 'classes' in data:
            detection_params['classes'] = [int(c) for c in data['classes']]

        if 'is_running' in data:
            detection_params['is_running'] = data['is_running']

        return jsonify({'status': 'success', 'params': detection_params.copy()})


@app.route('/upload_video', methods=['POST'])
def upload_video():
    """上传视频文件"""
    if 'video' not in request.files:
        return jsonify({'status': 'error', 'message': '没有上传文件'})

    file = request.files['video']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': '没有选择文件'})

    # 保存上传的视频
    upload_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
    os.makedirs(upload_dir, exist_ok=True)

    video_path = os.path.join(upload_dir, file.filename)
    file.save(video_path)

    with params_lock:
        detection_params['source_type'] = 'video'
        detection_params['video_path'] = video_path
        detection_params['is_running'] = True

    # 释放旧的视频源并标记需要新源
    release_video_source()
    need_new_source.set()

    return jsonify({
        'status': 'success',
        'message': '视频上传成功',
        'filename': file.filename
    })


@app.route('/start_camera', methods=['POST'])
def start_camera():
    """启动摄像头检测"""
    with params_lock:
        detection_params['source_type'] = 'camera'
        detection_params['video_path'] = None
        detection_params['is_running'] = True

    # 释放旧的视频源并标记需要新源
    release_video_source()
    need_new_source.set()

    return jsonify({'status': 'success', 'message': '摄像头已启动'})


@app.route('/stop_detection', methods=['POST'])
def stop_detection():
    """停止检测"""
    with params_lock:
        detection_params['is_running'] = False

    # 释放视频源
    release_video_source()

    return jsonify({'status': 'success', 'message': '检测已停止'})


@app.route('/get_stats')
def get_stats():
    """获取检测统计信息"""
    with params_lock:
        return jsonify({
            'fps': round(stats['fps'], 1),
            'total_detections': stats['total_detections'],
            'class_counts': dict(stats['class_counts']),
            'frame_count': stats['frame_count']
        })


@app.route('/get_params')
def get_params():
    """获取当前参数"""
    with params_lock:
        return jsonify({
            'conf': detection_params['conf'],
            'iou': detection_params['iou'],
            'classes': detection_params['classes'],
            'is_running': detection_params['is_running']
        })


if __name__ == '__main__':
    print("\n" + "="*50)
    print("空中障碍物实时检测系统")
    print("="*50)
    print(f"模型路径: {MODEL_PATH}")
    print(f"检测类别: {list(CLASS_NAMES.values())}")
    print("="*50 + "\n")

    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
