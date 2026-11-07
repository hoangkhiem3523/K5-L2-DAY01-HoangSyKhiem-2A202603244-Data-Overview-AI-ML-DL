# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 07/11/2026

**Runtime Colab:** GPU (NVIDIA GeForce RTX 3050 Laptop GPU)

**Python / PyTorch / Ultralytics:** Python 3.11 / PyTorch 2.6.0+cu124 / Ultralytics 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510501 ; `taxonomy_name`=ImageNet-1K
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
- Record này nói về **cả bức ảnh** (Image-level classification). Bởi vì trong tác vụ phân loại ảnh (image classification), mô hình chỉ gán một nhãn tổng thể duy nhất cho toàn bộ khung hình mà không cung cấp thông tin vị trí hay tọa độ (không có bounding box hay polygon mask) của từng đối tượng cụ thể trong ảnh.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
- Danh sách tên lớp đi kèm **taxonomy/checkpoint** cố định được thiết lập từ lúc huấn luyện (ở đây là bộ taxonomy ImageNet-1K gồm 1.000 lớp đi kèm file checkpoint `yolo11n-cls.pt`). Mô hình không thể tự nghĩ ra lớp mới nằm ngoài taxonomy đã được định nghĩa.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
- Vì `class_id` giúp hệ thống máy tính xử lý và lập chỉ mục chính xác, tránh sai lệch do lỗi chuỗi ký tự; `class_name` giúp con người đọc hiểu trực quan; còn `taxonomy_name` (ví dụ ImageNet-1K hay COCO-80) xác định hệ quy chiếu chuẩn của nhãn, bởi vì cùng một tên lớp (ví dụ "car") có thể có ID khác nhau hoặc phạm vi định nghĩa ngữ nghĩa khác nhau giữa các bộ taxonomy khác nhau.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
- Guideline cần quy định rõ **tiêu chí ưu tiên chọn chủ thể chính (salient object)**. Ví dụ: (1) Ưu tiên đối tượng nằm ở vùng trung tâm và chiếm diện tích lớn nhất trong khung hình; (2) Quy định thứ bậc ưu tiên giữa các loại nhãn (chẳng hạn nếu ảnh có nhiều phương tiện giao thông thì ưu tiên nhãn bối cảnh giao thông tổng thể thay vì một chiếc xe riêng lẻ); (3) Quy định rõ ngưỡng diện tích hoặc chuyển đổi sang tác vụ đa nhãn (multi-label) nếu không thể chọn được một nhãn duy nhất.
6. Vì sao không được lấy `score` của model làm ground truth?
- Vì `score` (model confidence score) chỉ là giá trị xác suất toán học thể hiện mức độ tự tin của mô hình dựa trên các đặc trưng học được, không phản ánh tính đúng đắn thực tế khách quan. Mô hình hoàn toàn có thể bị thiên kiến hoặc tự tin thái quá (overconfident) khi đưa ra dự đoán sai. Ground truth phải là sự thật chuẩn xác do con người kiểm chứng và dán nhãn theo đúng guideline.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=person ; `score`=0.912615 ; `bbox_xyxy`=[385.34, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
- Hộp nhận diện người phụ nữ bên phải có góc trên-trái nằm tại tọa độ pixel (x=385.34, y=69.24) và góc dưới-phải nằm tại tọa độ pixel (x=498.92, y=348.92) tính từ gốc tọa độ (0, 0) ở góc trên-cùng bên trái bức ảnh. Hộp có chiều rộng là 113.58 pixel và chiều cao là 279.68 pixel.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
- Ngưỡng `0.20`: Có **17** prediction (bắt thêm nhiều vật nhỏ như spoon, dining table, bottle, potted plant).
- Ngưỡng `0.60`: Có **6** prediction (chỉ giữ lại các vật thể rất rõ nét và chắc chắn gồm: person, bowl, oven).
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
- Reviewer phải tốn nhiều thời gian và công sức hơn để rà soát thủ công, loại bỏ các hộp rác/dương tính giả (false positives), xử lý các hộp bị trùng lặp hoặc gán sai nhãn do mô hình thiếu tự tin gây ra.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
- **Quy tắc viền khít (Tight Bounding Box):** "Hộp chữ nhật phải bao trọn toàn bộ các phần nhìn thấy được của vật thể; bốn cạnh của hộp phải tiếp xúc sát nhất với các điểm cực biên ngoài cùng của vật thể, không chừa khoảng trống quá 2 pixel ra ngoài nền và tuyệt đối không cắt lẹm vào chi tiết của đối tượng."
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
- **Chỉ ra vật thể:** Chiếc lò nướng (`oven`) ở mép góc trái dưới của ảnh bị cắt mép viền ảnh và người phụ nữ đứng phía xa bị bàn bếp che khuất phần thân dưới.
- **Cách xử lý:** Tôi sẽ viết thành **quy tắc trong guideline** thay vì escalation. Vì hiện tượng che khuất một phần (occlusion) và cắt mép ảnh (truncation) xuất hiện liên tục trong đời sống thực tế. Guideline cần quy định rõ: "Nếu phần nhìn thấy của vật thể vẫn nhận diện được rõ ràng và chiếm từ 20% diện tích trở lên, annotator vẽ bounding box bao trọn phần nhìn thấy đó. Nếu vật thể bị che khuất quá mức hoặc không thể xác định loại vật thể thì bỏ qua". Chỉ khi gặp trường hợp cực đoan ngoài guideline mới cần escalation.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=kitchen-001 ; `class_name`=person ; `score`=0.899262 ; số điểm polygon=`polygon_point_count`=350 ; 3–5 điểm đầu của `polygon_xy`=[[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
- Polygon cung cấp thông tin hình học chính xác ở cấp độ pixel về đường viền bao quanh (contour/shape), độ cong và diện tích thực tế của vật thể. Khác với box chỉ là khung chữ nhật bao quát chứa nhiều vùng nền trống xung quanh (background noise), polygon ôm khít đường biên và loại bỏ hoàn toàn các khoảng trống (ví dụ khoảng hở giữa cánh tay và thân mình).
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
- `instance_id` dùng để phân biệt từng cá thể/vật thể độc lập trong cùng một bức ảnh (ví dụ phân biệt người thứ nhất với người thứ hai dù cả hai cùng chung một lớp `person`). Nó không phải `class_id` vì class_id chỉ định danh thể loại đối tượng; nó cũng không phải `tracking_id` vì tracking_id dùng để liên kết cùng một vật thể xuyên suốt nhiều frame video theo thời gian, còn `instance_id` ở đây chỉ có phạm vi duy nhất trong một ảnh tĩnh của bài lab.
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
- **Quy tắc vẽ biên đa giác mượt (Polygon Boundary Rule):** "Vẽ chuỗi điểm đa giác khép kín ôm sát đường viền tự nhiên của vật thể; các góc cua và đường cong mềm (như vai, đầu gối) phải dùng mật độ điểm dày để đường viền không bị gãy khúc thô ráp; tuyệt đối không để mặt nạ lấn sang vật thể khác hoặc dính mảng nền lớn hơn 3 pixel."
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
- **Chỉ ra:** Vị trí đáy của chiếc bát (`bowl`) chạm vào mặt bàn bếp gỗ, nơi có bóng đổ tối khiến ranh giới tiếp xúc giữa đáy bát và mặt bàn bị mờ nhạt khó phân biệt.
- **Cách xử lý:** Tôi sẽ dùng **guideline** nếu quy tắc vẽ ranh giới tiếp xúc đã được nêu (ví dụ: vẽ theo hình elip tự nhiên của bát, bỏ qua bóng đổ). Tuy nhiên, nếu vùng bóng đổ và mặt bàn bị mất hoàn toàn độ tương phản và không thể xác định đáy bát nằm ở đâu ngay cả khi zoom tối đa, trường hợp này cần **escalation** lên Lab Coach để làm rõ quy chuẩn dán nhãn chuẩn hóa cho toàn đội ngũ gán nhãn, tránh việc mỗi người đoán mò một kiểu.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn phân loại duy nhất (`class_id`, `class_name`) cho toàn bộ bức ảnh | Ảnh `traffic` có nhiều loại xe (xe buýt, taxi, ô tô con), rất khó chọn một nhãn đại diện duy nhất nếu không có guideline rõ | Quan sát tổng thể ảnh, áp dụng quy tắc ưu tiên trong guideline để chọn đúng 1 nhãn chính | Kiểm tra xem nhãn được chọn có vi phạm thứ tự ưu tiên hoặc sai lệch chủ thể chính không |
| Phát hiện vật thể | Danh sách các hộp chữ nhật tọa độ `[x_min, y_min, x_max, y_max]` kèm `class_id` cho từng vật | Các vật thể nhỏ ở hậu cảnh (thìa, chai, lọ) hoặc vật bị che/cắt góc (lò nướng mép trái) dễ bị bỏ sót | Rà quét toàn bộ ảnh, vẽ khung chữ nhật bao khít từng đối tượng đạt chuẩn kích thước | Kiểm tra độ khít của box, phát hiện vật thể bị bỏ sót (False Negative) hoặc vẽ nhầm vùng nền (False Positive) |
| Instance segmentation | Danh sách các chuỗi tọa độ đa giác khép kín `polygon_xy` kèm `instance_id` và `class_id` | Đáy bát chạm mặt bàn bị bóng đổ làm mờ biên; viền người bị quần áo tối màu tiệp vào tủ bếp | Phóng to từng đối tượng, chấm các điểm đa giác bám sát từng pixel đường viền đối tượng | Soi kỹ độ mịn của đường biên viền, kiểm tra mặt nạ có bị lấn sang nền/vật thể khác hoặc tách nhầm instance không |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
- Chỉ sử dụng các tập ảnh mẫu COCO công khai và mô hình nguồn mở trong bài lab; tuyệt đối không tải lên hoặc công khai bất kỳ hình ảnh cá nhân, thông tin định danh hay dữ liệu nội bộ nào.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
- Tôi sẽ lập tức dừng mọi thao tác xử lý, giữ nguyên trạng dữ liệu và báo cáo ngay cho Giảng viên / Lab Coach phụ trách lớp học.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp **link repository** trên LMS
