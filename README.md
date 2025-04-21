## 📂 Dataset
- Dữ liệu được thu thập và xử lý thông qua [Roboflow](https://roboflow.com/), bao gồm các tư thế Yoga phổ biến được gán nhãn thủ công và chuẩn hóa cho bài toán nhận dạng động tác.

---

## 🧠 Nhận xét về kiến trúc hiện tại

Hiện tại mô hình sử dụng:
- **YOLOv8** để phát hiện người trong ảnh.
- **MediaPipe** để trích xuất các keypoint cơ thể.
- **MobileNetV2** để phân loại tư thế Yoga dựa trên ảnh hoặc keypoint.

➡️ Tuy nhiên, **kiến trúc này chưa tối ưu** vì sử dụng nhiều mô hình cùng lúc khiến hệ thống trở nên nặng, khó triển khai trên thiết bị có cấu hình thấp (như máy tính cá nhân, điện thoại, hoặc Raspberry Pi).

---

## 🚀 Đề xuất hướng tối ưu hoá

**Có hai hướng cải tiến đơn giản hơn nhưng vẫn hiệu quả:**

### 🔁 1. Dùng **YOLOv8 Pose** thay thế hoàn toàn YOLOv8 + MediaPipe
- YOLOv8 Pose có thể phát hiện người và đồng thời lấy được keypoints.
- Giảm số lượng model cần chạy → gọn hơn và nhanh hơn.

### 🧩 2. Dùng **MediaPipe Pose** duy nhất
- MediaPipe đã có thể nhận diện người và trích xuất keypoints rất chính xác trong video hoặc ảnh đơn.
- Chạy rất nhanh, nhẹ, dễ tích hợp vào ứng dụng real-time.
- Không cần dùng YOLO nữa nếu đầu vào không đông người.

---

✅ Việc lựa chọn hướng nào phụ thuộc vào mục tiêu ứng dụng:
- Nếu cần xử lý video đông người → ưu tiên YOLOv8 Pose.
- Nếu xử lý người đơn lẻ, video rõ ràng → MediaPipe là lựa chọn gọn nhẹ nhất.

