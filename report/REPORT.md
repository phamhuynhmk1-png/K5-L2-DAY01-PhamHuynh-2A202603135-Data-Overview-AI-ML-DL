# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06/11/2026

**Runtime Colab:** <điền CPU hoặc GPU>

**Python / PyTorch / Ultralytics:** <điền> / <điền> / 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35`

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

1. Chép record hạng 1: `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?

   Nó nói về **cả bức ảnh**. Record chỉ có `rank`, `class_id`, `class_name`, `score`, không có tọa độ box hay polygon nên không chỉ ra vật nào nằm ở đâu. Mỗi ảnh chỉ nhận một danh sách xếp hạng các lớp cho toàn bộ ảnh.

3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?

   Danh sách lớp đi kèm taxonomy và checkpoint (ở đây là ImageNet-1K với 1000 lớp cố định), không phải do model tự nghĩ ra. Model chỉ chọn trong danh sách có sẵn đó.

4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?

   `class_id` chỉ có nghĩa trong một taxonomy cụ thể: cùng một số ở taxonomy khác có thể là lớp khác. `class_name` giúp người đọc hiểu ngay, còn id giúp máy xử lý chính xác. `taxonomy_name` cho biết id và tên thuộc bảng nào (ImageNet-1K hay COCO-80), để người khác kiểm tra và tái hiện được kết quả.

5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?

   Guideline phải nói rõ quy tắc chọn **một** lớp: chọn theo chủ thể chiếm diện tích lớn nhất hay nổi bật nhất, khi có nhiều chủ thể ngang nhau thì xử lý thế nào (ưu tiên theo thứ tự, dùng lớp "khác/không rõ", hay hỏi Lab Coach). Ví dụ ảnh `traffic` có nhiều xe buýt, ô tô con, người đi bộ, người đi xe đạp. Nếu không có quy tắc thì mỗi annotator sẽ chọn một lớp khác nhau.

6. Vì sao không được lấy `score` của model làm ground truth?

   Score chỉ là mức tự tin của model, không phải đúng sai. Ở ảnh `traffic`, top-1 là `cab` (0.51) trong khi vật nổi bật nhất trong ảnh là xe buýt; top-5 còn lại là minibus (0.16), police_van (0.09), recreational_vehicle (0.05), streetcar (0.05), nên model chưa chắc chắn. Ground truth là nhãn do người gán theo guideline, nên phải do người quyết định chứ không theo score.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=oven ; `score`=0.6343 ; `bbox_xyxy`=[489.09, 200.73, 617.81, 344.88] ; `bbox_width`=128.72 ; `bbox_height`=144.15
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?

   Ảnh `kitchen` rộng 640 và cao 427 pixel, gốc (0, 0) ở góc trên-trái. Góc trên-trái của box ở (489.09, 200.73), tức cách mép trái khoảng 489 px và cách mép trên khoảng 201 px. Góc dưới-phải ở (617.81, 344.88). Chiều rộng = 617.81 − 489.09 = 128.72 px, chiều cao = 344.88 − 200.73 = 144.15 px, khớp với `bbox_width` và `bbox_height`.

3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?

   - Ngưỡng 0.20: **17** prediction (person ×2, bowl ×6, oven ×2, cup ×2, spoon ×3, potted plant, dining table, bottle).
   - Ngưỡng 0.60: **6** prediction (person ×2, bowl ×2, oven ×2).

4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?

   Từ 0.60 xuống 0.20, số prediction tăng từ 6 lên 17, nghĩa là reviewer phải kiểm tra thêm 11 box, nhiều box là vật nhỏ hoặc khó nhận như spoon, bottle, potted plant, dining table. Với mỗi box phải xác nhận vật có thật không, box có khít không và tên lớp có đúng không. Hạ ngưỡng giúp ít bỏ sót hơn nhưng tăng công kiểm tra và nguy cơ giữ lại box sai. Ngưỡng chỉ lọc **prediction** của model, không phải quy tắc để bỏ vật khi tạo ground truth: một cái thìa thật vẫn phải được gán nhãn dù model chỉ tự tin 0.2.

5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).

   Mỗi vật một box, ôm sát phần nhìn thấy của vật, không chừa nhiều nền. Các vật cùng lớp đứng cạnh nhau thì mỗi vật một box riêng. Box không được vượt ra ngoài ảnh.

6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?

   Trên `detection_predictions.png`, box `person 0.61` ở góc dưới bên trái chỉ bao một bàn tay/cẳng tay thò vào từ mép trái ảnh; phần còn lại của người nằm ngoài khung hình. Trong JSON, box này là `[0.12, 263.18, 61.55, 310.78]`, tức `x_min` gần bằng 0 và cao chỉ khoảng 48 px.

   - **Viết thành quy tắc guideline** cho phần vật bị cắt ở mép ảnh, vì tình huống này lặp lại ở rất nhiều ảnh: "Vật bị cắt ở mép ảnh thì vẽ box ôm phần nhìn thấy và dừng ở mép ảnh."
   - **Escalation hỏi Lab Coach** cho câu hỏi một bàn tay lẻ có đáng gán nhãn `person` không, và phần nhìn thấy tối thiểu là bao nhiêu thì mới gán. Guideline hiện chưa nói rõ điều này nên không tự quyết định.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=kitchen-002 ; `class_name`=bowl ; `score`=0.735743 ; số điểm polygon=`polygon_point_count`=67 ; 3–5 điểm đầu của `polygon_xy`=[53.0, 344.0], [52.0, 345.0], [50.0, 345.0], [49.0, 346.0], [48.0, 346.0]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?

   Box (`bbox_xyxy` = [31.44, 343.77, 100.89, 385.55]) chỉ là hình chữ nhật 69.45 × 41.78 px quanh cái bát. Polygon 67 điểm đi dọc theo đường viền cong của cái bát, nên cho biết hình dạng thật và vùng pixel thuộc vật, loại bỏ phần nền ở các góc của hình chữ nhật.

3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?

   `instance_id` (ví dụ `kitchen-002`) là mã của từng vật trong file output, giúp phân biệt các vật cùng lớp: bốn cái `bowl` trong `kitchen` có `instance_id` khác nhau là `kitchen-002`, `-003`, `-006`, `-010`. Nó **không phải class ID** (mã của loại, ở đây `class_id=45` mới là bowl) và **không phải tracking ID** (mã theo dõi một vật qua nhiều khung hình video). Nó chỉ có nghĩa trong file output này.

4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.

   Mỗi vật một polygon riêng, đi theo biên của phần vật nhìn thấy, không lấy nền. Hai vật cùng lớp đứng sát nhau thì tách thành hai polygon, không gộp. Các điểm polygon không được ra ngoài ảnh.

5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?

   Trên `segmentation_prediction.png`, `dining table 0.58` (`kitchen-005`, 598 điểm, box `[0.97, 234.74, 345.86, 423.87]`) tô cả mặt bàn gỗ lẫn các vật đặt trên đó (bột, lọ mứt, bát) và chồng lên polygon của `oven` và `bowl`, nên khó biết đâu là biên thật.

   - **Viết thành quy tắc guideline:** mỗi pixel chỉ thuộc một instance; vật có nhãn riêng nằm trên bàn thì cắt ra khỏi polygon của bàn, vì tình huống vật đặt trên bàn lặp lại ở nhiều ảnh.
   - **Escalation hỏi Lab Coach:** bàn làm bếp có nên gán lớp `dining table` hay không, vì taxonomy COCO-80 không có lớp "bàn bếp/bàn làm bánh" và guideline chưa nói rõ.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | 1 lớp cho cả ảnh (`class_id`, `class_name`, `taxonomy_name`) | `traffic`: top-1 `cab` (0.51) nhưng ảnh chủ yếu là xe buýt | Chọn một lớp theo quy tắc chọn chủ thể trong guideline | Lớp có khớp chủ thể chính và đúng guideline không |
| Phát hiện vật thể | 1 box `bbox_xyxy` + 1 lớp cho mỗi vật | `kitchen`: `person 0.61` chỉ là bàn tay ở mép trái; các box `bowl`/`cup`/`oven` chồng lấn ở góc dưới trái | Vẽ box ôm sát từng vật nhìn thấy và gán lớp | Có thiếu hoặc thừa box không, box có khít không, lớp có đúng không |
| Instance segmentation | 1 polygon `polygon_xy` + 1 lớp cho mỗi instance (`instance_id` riêng) | `kitchen-005` `dining table` tô cả vật đặt trên bàn; `potted plant` thực ra là bó lá treo | Vẽ polygon theo biên từng vật, tách instance cùng lớp | Polygon có bám biên, có tách đúng instance, có tự cắt nhau không |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)

   Chỉ dùng ba ảnh COCO công khai do bài lab cung cấp, không tải ảnh cá nhân hay dữ liệu nội bộ lên Colab hay GitHub, và không ghi họ tên, MSSV, email, số điện thoại vào báo cáo hay file output.

2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?

   Dừng lại, không dùng tiếp dữ liệu đó, và báo cho Lab Coach / giảng viên để được hướng dẫn.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
