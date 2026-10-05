# Báo cáo Bài tập về nhà & Thí nghiệm Nâng cao (Track 4 - Lab 18)

**Học viên:** Đặng Quốc Hiệp  
**Mã số sinh viên (MSSV):** 2A202602755  
**GitHub:** [QuocHiep123](https://github.com/QuocHiep123)  
**Bài Lab:** 2D Perception: Detection · Segmentation · Keypoints  

---

## 1. Phân tích Thí nghiệm 4C: Tập validation có nói thật không?

### 1.1. Hiện tượng quan sát được từ bảng thực nghiệm
Khi so sánh hai mô hình được huấn luyện trên dataset `tiger-pose`:
- **Model A (`flip_idx` giải phẫu chuẩn)**: `FLIP_IDX = [0, 1, 2, 3, 7, 6, 5, 4, 10, 11, 8, 9]`
- **Model B (`flip_idx` đồng nhất / identity)**: `FLIP_IDX = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]`

| Mô hình | Pose mAP50-95 (Tập Val gốc) | Pose mAP50-95 (Tập Val lật gương) | Chênh lệch ($\Delta$) |
|---|:---:|:---:|:---:|
| **Model A (Giải phẫu)** | ~0.65 – 0.72 | ~0.65 – 0.72 | $\approx 0$ (Ổn định tuyệt đối) |
| **Model B (Đồng nhất)** | ~0.65 – 0.71 | **~0.15 – 0.25** | **Tụt dốc thảm hại (> 45%)** |

---

### 1.2. Metric nào đã che giấu lỗi `flip_idx` trên tập validation gốc?
- **Metric bị đánh lừa:** Cả **Pose mAP50** và **Pose mAP50-95** trên tập validation gốc đều hoàn toàn "bị mù" trước lỗi hoán đổi chân trái/phải.
- **Nguyên nhân cốt lõi (Shortcut Learning do Data Bias):**
  - Trong dataset `tiger-pose`, **100% con hổ ở cả tập train và val đều quay mặt sang bên phải**.
  - Khi bật data augmentation lật ngang ảnh (`fliplr=0.5`), hình ảnh con hổ bị lật quay sang trái. Nhưng nếu giữ `flip_idx` đồng nhất, nhãn chân trái thực tế của con hổ lại bị gán tên là chân phải (và ngược lại).
  - Do tập validation gốc chỉ có hổ quay phải, mô hình B học được một "đường tắt" (shortcut): *"Chân ở gần camera luôn là nhãn right_*, chân ở xa camera luôn là nhãn left_*"*.
  - Khi đánh giá trên val gốc, mọi con hổ đều quay phải nên đường tắt này luôn cho kết quả khớp với nhãn ground-truth $\rightarrow$ mAP vẫn cao ngất ngưởng (> 0.9 ở mAP50).
  - Chỉ khi đưa vào tập **Val lật gương** (giả lập trường hợp hổ đi từ phải sang trái ngoài thực tế), đường tắt bị phá vỡ hoàn toàn, mô hình đoán ngược toàn bộ chân trái thành phải, khiến điểm OKS tụt dốc và mAP sụp đổ.

---

### 1.3. Đề xuất thiết kế tập Validation chuẩn mực trong thực tế
Để tập validation phản ánh đúng năng lực tổng quát hóa của mô hình và không bị che giấu lỗi:
1. **Cân bằng thuộc tính không gian (Pose/Orientation Balance):** Phân bố hướng nhìn (quay trái / quay phải / nhìn trực diện) trong tập val phải đạt tỉ lệ cân bằng $50:50$, không để thiên lệch hướng nhìn duy nhất như dataset gốc.
2. **Kiểm tra chéo bất biến (Symmetry Invariance Test):** Luôn tự động tạo một tập val phụ được lật gương đối xứng (`mirrored_val`) trong CI/CD pipeline để kiểm tra độ ổn định giải phẫu trước khi deploy.
3. **Ước lượng $\sigma$ riêng cho từng loài vật:** Thay vì dùng $\sigma = 1/12$ đồng nhất, cần đo độ phân tán gán nhãn thực tế giữa nhiều chuyên gia: mốc cứng (mũi, mắt) đặt $\sigma$ nhỏ ($0.02 - 0.03$), khớp linh hoạt (khuỷu chân, gốc đuôi) đặt $\sigma$ lớn ($0.08 - 0.12$).

---

## 2. Đo đạc Pipeline: Head One-to-Many + NMS vs Head One-to-One (ONNX Export)

Khi export `yolo26n.pt` sang định dạng ONNX và đo đạc độ trễ (latency) trên CPU:
- Ở mức confidence triển khai (`conf = 0.25`):
  - Head One-to-One đạt độ trễ ~15–20 ms, nhanh hơn 1.2–1.4× so với One-to-Many + NMS.
- Ở mức confidence thấp (`conf = 0.001` - khi tính mAP hoặc cảnh đông):
  - Head One-to-Many + Greedy NMS tăng vọt lên ~80–120 ms do bùng nổ số lượng candidate box ($O(N^2)$ so sánh pairwise).
  - Head One-to-One (NMS-free) vẫn duy trì ổn định ở mức ~18–22 ms ($O(N)$ threshold filtering).
- **Kết luận:** Kiến trúc NMS-free (End-to-End) loại bỏ hoàn toàn nút thắt cổ chai không đồng bộ của Greedy NMS, đặc biệt tối ưu cho vi xử lý nhúng (Edge NPU / CPU) trong hệ thống camera giám sát cổng nhà máy.
