const videoElement = document.getElementById('video');

// 获取用户媒体设备的访问权限
navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        videoElement.srcObject = stream;
    })
    .catch(err => {
        console.error('Error accessing media devices.', err);
    });

// 创建 WebSocket 连接
const socket = new WebSocket('ws://localhost:8000/video_ws');

// 当 WebSocket 连接打开时
socket.addEventListener('open', () => {
    console.log('WebSocket connection established.');

    // 每秒抓取视频帧并发送给服务端
    setInterval(() => {
        captureFrameAndSend();
    }, 1000); // 1000ms = 1s
});

// 捕获当前视频帧并发送给服务端
function captureFrameAndSend() {
    if (videoElement.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
        const canvas = document.createElement('canvas');
        canvas.width = videoElement.videoWidth;
        canvas.height = videoElement.videoHeight;
        const context = canvas.getContext('2d');

        context.drawImage(videoElement, 0, 0, canvas.width, canvas.height);

        canvas.toBlob(blob => {
        console.log(blob)
            if (blob) {
                if (socket.readyState === WebSocket.OPEN) {
                    socket.send(blob);
                } else {
                    console.error('WebSocket is not open. Unable to send data.');
                }
            } else {
                console.error('Failed to convert canvas to blob.');
            }
        }, 'image/jpeg');
    } else {
        console.error('Video is not ready for capturing.');
    }
}


// 处理WebSocket连接关闭的情况
socket.addEventListener('close', () => {
    console.log('WebSocket connection closed.');
});

// 处理WebSocket错误
socket.addEventListener('error', error => {
    console.error('WebSocket error:', error);
});
