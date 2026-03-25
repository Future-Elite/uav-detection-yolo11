/**
 * 空中障碍物实时检测系统 - 前端交互逻辑
 */

// DOM元素
const elements = {
    videoFeed: document.getElementById('videoFeed'),
    placeholder: document.getElementById('placeholder'),
    startCamera: document.getElementById('startCamera'),
    stopDetection: document.getElementById('stopDetection'),
    videoUpload: document.getElementById('videoUpload'),
    videoName: document.getElementById('videoName'),
    confThreshold: document.getElementById('confThreshold'),
    confValue: document.getElementById('confValue'),
    iouThreshold: document.getElementById('iouThreshold'),
    iouValue: document.getElementById('iouValue'),
    fpsValue: document.getElementById('fpsValue'),
    totalDetections: document.getElementById('totalDetections'),
    classCheckboxes: document.querySelectorAll('input[name="class"]')
};

// 统计更新定时器
let statsInterval = null;

// 当前状态
let isRunning = false;

/**
 * 更新参数到后端
 */
async function updateParams() {
    const conf = parseFloat(elements.confThreshold.value);
    const iou = parseFloat(elements.iouThreshold.value);
    const classes = Array.from(elements.classCheckboxes)
        .filter(cb => cb.checked)
        .map(cb => parseInt(cb.value));

    try {
        const response = await fetch('/update_params', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                conf: conf,
                iou: iou,
                classes: classes
            })
        });

        const data = await response.json();
        console.log('参数已更新:', data);
    } catch (error) {
        console.error('更新参数失败:', error);
    }
}

/**
 * 启动摄像头检测
 */
async function startCamera() {
    try {
        const response = await fetch('/start_camera', {
            method: 'POST'
        });

        const data = await response.json();

        if (data.status === 'success') {
            startVideoFeed();
            showNotification('摄像头已启动', 'success');
        } else {
            showNotification('启动摄像头失败: ' + data.message, 'error');
        }
    } catch (error) {
        console.error('启动摄像头失败:', error);
        showNotification('启动摄像头失败', 'error');
    }
}

/**
 * 停止检测
 */
async function stopDetection() {
    try {
        const response = await fetch('/stop_detection', {
            method: 'POST'
        });

        const data = await response.json();

        if (data.status === 'success') {
            stopVideoFeed();
            showNotification('检测已停止', 'info');
        }
    } catch (error) {
        console.error('停止检测失败:', error);
    }
}

/**
 * 上传视频文件
 */
async function uploadVideo(file) {
    const formData = new FormData();
    formData.append('video', file);

    try {
        showNotification('正在上传视频...', 'info');

        const response = await fetch('/upload_video', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.status === 'success') {
            elements.videoName.textContent = '当前视频: ' + data.filename;
            elements.videoName.classList.add('show');
            startVideoFeed();
            showNotification('视频上传成功', 'success');
        } else {
            showNotification('上传失败: ' + data.message, 'error');
        }
    } catch (error) {
        console.error('上传视频失败:', error);
        showNotification('上传视频失败', 'error');
    }
}

/**
 * 启动视频流
 */
function startVideoFeed() {
    isRunning = true;

    // 显示视频，隐藏占位符
    elements.videoFeed.src = '/video_feed?' + new Date().getTime();
    elements.videoFeed.classList.add('show');
    elements.placeholder.classList.add('hidden');

    // 启动统计更新
    if (statsInterval) {
        clearInterval(statsInterval);
    }
    statsInterval = setInterval(updateStats, 500);
}

/**
 * 停止视频流
 */
function stopVideoFeed() {
    isRunning = false;

    // 隐藏视频，显示占位符
    elements.videoFeed.src = '';
    elements.videoFeed.classList.remove('show');
    elements.placeholder.classList.remove('hidden');

    // 停止统计更新
    if (statsInterval) {
        clearInterval(statsInterval);
        statsInterval = null;
    }

    // 重置统计显示
    elements.fpsValue.textContent = '0';
    elements.totalDetections.textContent = '0';
    document.querySelectorAll('.class-count').forEach(el => {
        el.textContent = '0';
    });
}

/**
 * 更新统计信息
 */
async function updateStats() {
    if (!isRunning) return;

    try {
        const response = await fetch('/get_stats');
        const data = await response.json();

        elements.fpsValue.textContent = data.fps;
        elements.totalDetections.textContent = data.total_detections;

        // 更新各类别统计
        for (const [className, count] of Object.entries(data.class_counts)) {
            const countEl = document.getElementById('count-' + className);
            if (countEl) {
                countEl.textContent = count;
            }
        }
    } catch (error) {
        console.error('获取统计信息失败:', error);
    }
}

/**
 * 显示通知
 */
function showNotification(message, type = 'info') {
    // 创建通知元素
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;

    // 添加样式
    Object.assign(notification.style, {
        position: 'fixed',
        top: '20px',
        right: '20px',
        padding: '15px 25px',
        borderRadius: '8px',
        color: 'white',
        fontSize: '0.95rem',
        zIndex: '1000',
        animation: 'slideIn 0.3s ease',
        boxShadow: '0 4px 15px rgba(0,0,0,0.3)'
    });

    // 根据类型设置背景色
    const colors = {
        success: '#2ecc71',
        error: '#e74c3c',
        info: '#3498db'
    };
    notification.style.background = colors[type] || colors.info;

    document.body.appendChild(notification);

    // 3秒后移除
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => {
            notification.remove();
        }, 300);
    }, 3000);
}

// 添加动画样式
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

// 事件监听器
document.addEventListener('DOMContentLoaded', function() {
    // 置信度滑块
    elements.confThreshold.addEventListener('input', function() {
        elements.confValue.textContent = this.value;
    });

    elements.confThreshold.addEventListener('change', updateParams);

    // IoU滑块
    elements.iouThreshold.addEventListener('input', function() {
        elements.iouValue.textContent = this.value;
    });

    elements.iouThreshold.addEventListener('change', updateParams);

    // 类别复选框
    elements.classCheckboxes.forEach(checkbox => {
        checkbox.addEventListener('change', updateParams);
    });

    // 摄像头按钮
    elements.startCamera.addEventListener('click', startCamera);

    // 停止检测按钮
    elements.stopDetection.addEventListener('click', stopDetection);

    // 视频上传
    elements.videoUpload.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            uploadVideo(file);
        }
    });

    // 视频加载错误处理
    elements.videoFeed.addEventListener('error', function() {
        console.error('视频流加载失败');
        if (isRunning) {
            showNotification('视频流连接失败，正在重试...', 'error');
            // 尝试重新连接
            setTimeout(() => {
                if (isRunning) {
                    elements.videoFeed.src = '/video_feed?' + new Date().getTime();
                }
            }, 1000);
        }
    });

    console.log('空中障碍物实时检测系统已初始化');
});
