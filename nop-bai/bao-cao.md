# Báo Cáo Lab Day 21 - CI/CD cho AI Systems

| | |
|---|---|
| Họ và tên | Cao Minh Quang |
| MSSV | 2A202601884 |
| Lớp / Khóa | K4 |
| Repo GitHub | https://github.com/minhquang1604/TRACK2_Day21_2A202601884_CaoMinhQuang |
| Ngày nộp | 21/08/2026 |

---

## 1. Bộ Siêu Tham Số Đã Chọn và Lý Do

| Lần chạy | n_estimators | learning_rate | max_depth | f1_score | accuracy |
|---|---|---|---|---|---|
| 1 | 100 | 0.1 | 3 | 0.7109 | 0.8780 |
| 2 | 50 | 0.05 | 2 | 0.6051 | 0.8460 |
| 3 | 200 | 0.1 | 5 | 0.7149 | 0.8740 |
| 4 | 150 | 0.2 | 3 | 0.7091 | 0.8720 |
| 5 | 300 | 0.05 | 4 | 0.7070 | 0.8740 |

**Bộ siêu tham số đã chọn:** `n_estimators=200`, `learning_rate=0.1`, `max_depth=5`.

**Lý do:** Bộ này đạt f1_score cao nhất (0.7149), vượt ngưỡng 0.65. Đáng chú ý, lần chạy 1 có
accuracy cao nhất (0.878) nhưng f1_score thấp hơn lần 3 — chứng tỏ accuracy không phản ánh đúng
khả năng phát hiện lớp thiểu số. Lần chạy 2 (tham số nhỏ, learning_rate thấp) tụt xuống dưới
ngưỡng 0.65, cho thấy mô hình quá đơn giản với dữ liệu mất cân bằng. Đánh đổi quan sát được: tăng
max_depth cải thiện f1 rõ hơn so với chỉ tăng n_estimators hay learning_rate riêng lẻ.

---

## 2. Vì Sao Ngưỡng Chất Lượng Đặt Trên F1 Chứ Không Phải Accuracy

Tập dữ liệu Adult mất cân bằng: chỉ 24,8% mẫu thuộc lớp thu nhập cao. Một mô hình vô dụng, luôn
trả lời "thu nhập thấp", vẫn đạt accuracy 0,752 — con số gây hiểu nhầm vì mô hình không bắt được
trường hợp thu nhập cao nào (f1_score = 0). F1 của lớp dương là trung bình điều hòa giữa
precision và recall trên đúng lớp thiểu số, phản ánh trực tiếp khả năng phát hiện đối tượng thu
nhập cao — điều accuracy che giấu do bị lớp đa số áp đảo. Vì vậy lab đặt ngưỡng trên f1_score
(>= 0.65) thay vì accuracy. Cũng không dùng `average="weighted"`/`"macro"` vì các cách tính này
gộp cả điểm số của lớp đa số vào trung bình, kéo giá trị lên cao giả tạo, làm mất ý nghĩa "mô
hình có nhận diện được lớp thiểu số hay không".

---

## 3. Khó Khăn Gặp Phải và Cách Giải Quyết

| Khó khăn | Nguyên nhân | Cách giải quyết |
|---|---|---|
| `import mlflow` báo lỗi thiếu `pkg_resources` | mlflow 2.13.0 cần `pkg_resources`, bị loại khỏi setuptools bản 70+ | Ghim `setuptools<70` trong `requirements.txt` |
| `aws s3 mb` bị AccessDenied | IAM user chỉ có sẵn quyền EC2/IAM/VPC, không có quyền S3 | Tự cấp policy S3 tối thiểu, tạo riêng IAM user chỉ quyền S3 cho GitHub Actions |
| `git push` không tự kích hoạt Actions | Repo là fork, GitHub tắt workflow trên fork tới khi bật thủ công | Vào tab Actions bấm nút "enable workflows" một lần |

---

## 4. So Sánh Bước 2 và Bước 3 (bắt buộc, 2 - 3 câu)

| | f1_score | accuracy |
|---|---|---|
| Bước 2 (chỉ `train_batch1`, 22.361 mẫu) | 0.7149 | 0.8740 |
| Bước 3 (thêm `train_batch2`, 44.722 mẫu) | 0.7354 | 0.8820 |

**Nhận xét:** f1_score tăng nhẹ (+0,0205), accuracy cũng tăng (+0,008) sau khi gấp đôi dữ liệu.
Vì train_batch2 lấy mẫu ngẫu nhiên từ cùng nguồn Adult như train_batch1 (cùng phân phối), mức
tăng khiêm tốn này chủ yếu do mô hình quan sát thêm nhiều mẫu lớp thiểu số, giảm phương sai ước
lượng, chứ không phải do dữ liệu mới mang thêm thông tin/phân phối khác. Quan trọng hơn con số:
quy trình tự động đã chạy đúng — commit dữ liệu mới tự kích hoạt toàn bộ pipeline (Unit Test →
Train → Quality Gate → Release) không cần thao tác thủ công.

---

## 5. Phần Bonus Đã Thực Hiện

- [x] Bonus 1 - DagsHub: MLflow tracking chuyển sang server DagsHub trong CI, xem tại https://dagshub.com/minhquang1604/TRACK2_Day21_2A202601884_CaoMinhQuang.mlflow
- [x] Bonus 2 - Ngưỡng tối ưu 0.30 cho f1=0.7537, cao hơn ngưỡng mặc định 0.5 (f1=0.7354).
- [x] Bonus 3 - `outputs/detail.txt` (confusion matrix + precision/recall/lớp) lưu làm CI artifact.
- [x] Bonus 4 - Job Rollback Guard: chặn promote nếu f1 mới thấp hơn model đang chạy trên S3.
- [x] Bonus 5 - Cảnh báo drift nếu tỷ lệ lớp dương lệch >5pp so với 24.8%; ghi `positive_ratio` vào report.json.
