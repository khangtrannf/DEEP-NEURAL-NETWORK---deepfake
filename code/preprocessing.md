# Phần 1 – Preprocess, Clean, Normalize, Augment

| | |
|---|---|
| **Môn** | Deep Neural Network |
| **Bài toán** | Deepfake Image Detection (phân loại ảnh Real vs Fake) |
| **Dataset** | https://www.kaggle.com/datasets/saurabhbagchi/deepfake-image-detection |
| **Người thực hiện** | Khang |
| **File code** | `dataset.py`, `dedup.py`, `inspect_data.py`, `visualize_augmentation.py` |

---

## 1. Tóm tắt

| Bước | Quyết định | Ghi chú ngắn |
|---|---|---|
| Clean | **Có** – loại ảnh trùng, loại ảnh mang 2 nhãn | Ban đầu dự đoán "không cần clean", nhưng kiểm tra dữ liệu thật thì có trùng (mục 3) |
| Resize | Có – 224×224 | Ảnh gốc kích thước rất khác nhau |
| Normalize | Có – mean/std ImageNet | Cần cho Transfer Learning ở phần 4 |
| Augment | **Có – chỉ tập Train** | Lật ngang, lật dọc, xoay ±15°, zoom 0.8×–1.2× |
| Chia tập | Gộp train+test gốc, chia 80/10/10 stratified, theo nhóm ảnh trùng | Có tuỳ chọn dùng test gốc |

Kết quả: **971 ảnh** sạch, chia thành **Train 776 / Val 97 / Test 98**, cả nhóm dùng chung qua một hàm `get_dataloaders()`.

---

## 2. Dataset

| Thư mục gốc | Real | Fake | Tổng |
|---|---|---|---|
| `train-...-001/train/` | 326 | 153 | 479 |
| `test-...-001/test/` | 110 | 389 | 499 |
| `Sample_fake_images/` | – | 5 | 5 (ảnh mẫu, **loại bỏ**, không thuộc train/test) |
| **Dùng được** | 436 | 542 | **978** |

Đặc điểm cần biết:
- Rất nhỏ (~980 ảnh) và đa dạng chủ đề: phong cảnh, động vật, tranh, người... Không phải chỉ mặt người. Bài toán thực chất là phân biệt ảnh do AI sinh ra với ảnh thật nói chung.
- Tỉ lệ Real/Fake của train gốc (68% real) và test gốc (78% fake) **lệch ngược nhau**.
- Kích thước ảnh rất khác nhau: chiều rộng từ ~180 đến 8495 px (trung vị ~1000); 78% ảnh Fake và 92% ảnh Real không vuông.

---

## 3. Kiểm tra dữ liệu (`inspect_data.py`)

Đã chạy trên dataset thật (978 ảnh):

| Mục kiểm tra | Kết quả | Xử lý |
|---|---|---|
| File ảnh hỏng | 0 | – |
| Chế độ màu | Chủ yếu RGB; có RGBA, P (palette), L (xám) ở cả 2 lớp | `.convert("RGB")` khi đọc ảnh |
| Định dạng | JPEG và PNG (PNG: 110 Fake, 33 Real) | Không xử lý, ghi nhận ở mục 10 |
| Ảnh trùng y hệt | 4 nhóm (cùng nội dung file) | Giữ 1 ảnh, bỏ bản còn lại |
| Ảnh gần trùng | ~41 nhóm (83 ảnh): cùng một hình nhưng khác độ phân giải (khác kích thước, cùng tỉ lệ khung hình). **27 nhóm nằm vắt ngang cả train và test gốc** | Giữ ảnh, nhưng cả nhóm luôn nằm cùng một tập |
| Một hình mang 2 nhãn | 1 nhóm (3 ảnh): vừa là Fake vừa là Real | Loại cả nhóm vì nhãn không đáng tin |

**Ý nghĩa của 27 nhóm vắt ngang train/test:** một phần tập test gốc của Kaggle thực chất đã xuất hiện trong tập train (ở độ phân giải khác). Nếu dùng nguyên train/test gốc, điểm test sẽ cao hơn thực tế (rò rỉ dữ liệu). Đây là lý do chính để làm sạch và chia lại.

---

## 4. Clean (`dedup.py`)

Hàm `deduplicate()` chạy 4 bước:

1. Bỏ file không đọc được (hiện tại: 0 file).
2. Bỏ ảnh **trùng y hệt** (so mã md5 của file), giữ ảnh đầu tiên. → bỏ 4 ảnh.
3. Gom ảnh **gần trùng** thành nhóm: hai ảnh được coi là một nhóm nếu "dhash" (vân tay 64 bit của ảnh, giống nhau khi ảnh trông giống nhau dù đổi kích thước) lệch ≤ 4 bit **và** tỉ lệ khung hình khớp nhau (sai khác < 0.02). Ảnh gần trùng **không bị xoá**, chỉ được gán chung mã nhóm.
4. Loại các nhóm có ảnh mang nhãn mâu thuẫn (Real lẫn Fake). → bỏ 1 nhóm, 3 ảnh.

Kết quả: **978 → 971 ảnh** (537 Fake, 434 Real).

Lưu ý:
- Số liệu trên tính từ file `images.csv` do `inspect_data.py` sinh ra. Khi chạy thật, cách tính hash nhanh hơn nên có thể lệch vài ảnh.
- Phát hiện ảnh gần trùng là phương pháp xấp xỉ, **chưa kiểm tra bằng mắt từng cặp**. Nếu báo nhầm hai ảnh khác nhau là một nhóm thì chỉ khiến hai ảnh đó nằm cùng một tập, không gây hại.
- Lần chạy đầu mất khoảng 1 phút để tính vân tay; kết quả được lưu cache (`.dedup_cache.json` trong thư mục kagglehub) nên các lần sau nhanh.

---

## 5. Chia Train / Val / Test

Chia **stratified** (mỗi lớp chia cùng tỉ lệ) và **theo nhóm** (các ảnh gần trùng luôn vào cùng một tập), `seed = 42`.

### Cách mặc định – `split_mode="pooled"`

Gộp train + test gốc rồi chia 80/10/10.

| Tập | Số ảnh | Fake | Real |
|---|---|---|---|
| Train | 776 | 429 (55%) | 347 |
| Val | 97 | 54 | 43 |
| Test | 98 | 54 | 44 |

### Cách thay thế – `split_mode="official"`

Giữ nguyên tập test gốc; val lấy 20% từ train; loại 26 ảnh train trùng với ảnh test để không rò rỉ.

| Tập | Số ảnh | Fake | Real |
|---|---|---|---|
| Train | 359 | 114 (32%) | 245 |
| Val | 90 | 29 | 61 |
| Test | 496 | 387 (78%) | 109 |

### Vì sao chọn "pooled" làm mặc định

- Train lớn gần gấp đôi (776 so với 359), rất quan trọng với dataset nhỏ.
- Ba tập cùng phân phối lớp (~55% Fake), nên so sánh công bằng giữa 3 phần (CNN đơn giản, CNN phức tạp, Transfer Learning).
- Với "official", train 32% Fake nhưng test 78% Fake: model học nhiều Real mà bị thi toàn Fake, nên accuracy dễ gây hiểu lầm.

Đánh đổi của "pooled": tự đổi cách chia so với bản gốc của Kaggle (kết quả không so trực tiếp được với người khác dùng chia gốc); Val/Test chỉ ~98 ảnh nên số liệu dao động (mỗi ảnh sai lệch ~1%).

Ghi chú chung:
- Mỗi tập là một `Dataset` độc lập với transform riêng → augmentation chắc chắn chỉ nằm ở Train.
- Nhãn lấy theo tên thư mục: **Fake = 0, Real = 1** (thứ tự alphabet, trả về trong `classes`).
- **Cả nhóm phải dùng cùng một `split_mode` và `seed`**, không ai tự chia lại.

---

## 6. Augmentation (chỉ áp dụng cho Train)

| Kỹ thuật | Hàm torchvision | Tham số | Ý nghĩa |
|---|---|---|---|
| Lật ngang | `RandomHorizontalFlip` | p = 0.5 | Đối tượng có thể quay trái hoặc phải |
| Lật dọc | `RandomVerticalFlip` | p = 0.5 | Thêm biến thể hướng ảnh |
| Xoay trái/phải | `RandomAffine(degrees=15)` | góc ngẫu nhiên trong [−15°, +15°] | Mô phỏng ảnh bị nghiêng |
| Zoom nhỏ / to | `RandomAffine(scale=(0.8, 1.2))` | 0.8× (zoom nhỏ) đến 1.2× (zoom to) | Mô phỏng đối tượng xa/gần |

Ghi chú:
- Xoay và zoom gộp trong **một** `RandomAffine` để ảnh chỉ bị nội suy 1 lần (ít mờ hơn so với xoay rồi zoom riêng). Về hiệu quả vẫn là xoay trái/phải + zoom to/nhỏ như yêu cầu của leader.
- Vùng trống do xoay / zoom nhỏ được tô đen.
- Các phép được random lại **mỗi lần ảnh được đọc**, nên mỗi epoch model thấy một biến thể khác của cùng một ảnh (số file không tăng, độ đa dạng tăng).
- **Không** dùng `ColorJitter` hay biến đổi màu/nhiễu: đề chỉ yêu cầu biến đổi hình học, và với bài này màu sắc/nhiễu ảnh có thể là dấu hiệu để phân biệt Real/Fake nên không muốn làm mất chúng.
- Val/Test **không** augment, để kết quả đánh giá ổn định và lặp lại được.

**Hình minh hoạ** (chạy `python visualize_augmentation.py`, ảnh lưu ở `augmentation_samples/`):

- `1_each_technique.png`: ảnh gốc và từng kỹ thuật áp dụng riêng lẻ. → *(chèn ảnh vào đây)*
- `2_random_pipeline.png`: ảnh gốc và 5 lần random từ pipeline Train thật. → *(chèn ảnh vào đây)*

---

## 7. Resize và Normalize

**Resize về 224×224.** Ảnh gốc có kích thước rất khác nhau nên bắt buộc đưa về cùng kích thước để gom batch; 224 là kích thước chuẩn của ConvNeXt và nhiều model pretrained. Cách làm hiện tại là bóp thẳng về 224×224 (đổi tỉ lệ khung hình) vì đơn giản nhất. Cách khác là `Resize(256) + CenterCrop(224)` (mất phần rìa ảnh) hoặc thêm viền đen (pad). **Chưa thử so sánh các cách này**; nếu muốn có thể đưa vào phần thử nghiệm sau.

**Normalize bằng mean/std của ImageNet** (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).
- Đây không phải bước "làm sạch" dữ liệu, mà là chuẩn hoá giá trị kênh màu sau `ToTensor`.
- Cần cho phần 4, vì ConvNeXt/EfficientNet pretrained được huấn luyện với phép chuẩn hoá này.
- Dùng chung cho cả 3 phần để so sánh công bằng.

---

## 8. Pipeline cuối cùng

```
Scan ảnh (bỏ Sample_*, nhãn theo thư mục real/fake)
  → Clean: bỏ file hỏng, bỏ trùng y hệt, gom nhóm gần trùng, bỏ nhóm mâu thuẫn nhãn
  → Chia Train/Val/Test (stratified + theo nhóm, seed=42)
  → Train: Resize 224 → H-flip → V-flip → RandomAffine(±15°, 0.8–1.2) → ToTensor → Normalize
  → Val/Test: Resize 224 → ToTensor → Normalize
  → DataLoader
```

---

## 9. Bàn giao cho phần 2, 3, 4

```python
from dataset import get_dataloaders

train_loader, val_loader, test_loader, classes = get_dataloaders(batch_size=32)
# classes = ['Fake', 'Real']  ->  Fake = 0, Real = 1
# images: (B, 3, 224, 224) float32 đã normalize | labels: (B,) int64
```

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `batch_size` | 32 | |
| `seed` | 42 | Cố định cách chia và thứ tự trộn |
| `split_mode` | `"pooled"` | `"pooled"`: gộp rồi chia 80/10/10. `"official"`: dùng test gốc |
| `image_size` | 224 | Đổi nếu model cần kích thước khác (ví dụ EfficientNet-B3 thường dùng ~300) |
| `num_workers` | 2 | Đặt `0` nếu máy yếu hoặc Windows báo lỗi DataLoader |
| `dataset_path` | `None` | Truyền đường dẫn nếu đã tải dataset sẵn, mặc định tự tải bằng kagglehub |
| `remove_duplicates` | `True` | Tắt chống trùng (không khuyến khích) |

Lưu ý cho các phần sau:
- Chỉ `train_loader` có augmentation.
- Báo cáo thêm **confusion matrix, F1, Precision, Recall**, không chỉ accuracy. Val/Test chỉ ~98 ảnh nên đừng kết luận khi hai model chỉ hơn kém nhau 1–2%.
- Phần 4 dùng trực tiếp được (đã normalize theo ImageNet).

---

## 10. Hạn chế cần nêu khi báo cáo

1. **Dataset nhỏ**: ~971 ảnh, Val/Test chỉ ~98 ảnh → số liệu dao động mạnh.
2. **Đa dạng chủ đề**: không chỉ mặt người, nên model phải phân biệt ảnh AI với ảnh thật nói chung.
3. **Có thể có nhiễu nhãn**: trong ảnh mẫu, vài ảnh Fake trông giống ảnh chụp màn hình hoặc infographic. Chưa kiểm tra toàn bộ.
4. **Có "đường tắt" về siêu dữ liệu (yếu)**: 77% ảnh PNG là Fake (tổng thể Fake chỉ 55%); kích thước 1024×1024 có 28 ảnh Fake so với 1 ảnh Real. Mô hình chỉ dùng siêu dữ liệu (kích thước, định dạng, dung lượng file) đạt khoảng 62% accuracy so với 55% khi đoán lớp đa số. Nghĩa là model có thể học một phần từ đặc điểm file chứ không chỉ từ nội dung ảnh.
5. **Ảnh gần trùng chưa kiểm tra bằng mắt**, chỉ dựa trên hash.
6. **Tự chia lại dataset** nên không so trực tiếp được với kết quả dùng cách chia gốc của Kaggle.


---

## 12. Cách chạy

```bash
pip install -r requirements.txt

python inspect_data.py             # kiểm tra dữ liệu: file hỏng, trùng, kích thước (sinh thư mục inspect_output/)
python dataset.py                  # tải dataset, làm sạch, chia tập, in số ảnh và shape 1 batch
python visualize_augmentation.py   # sinh hình minh hoạ augmentation (thư mục augmentation_samples/)
```

| File | Vai trò |
|---|---|
| `dataset.py` | Pipeline chính: quét ảnh, chia tập, transform, DataLoader |
| `dedup.py` | Phát hiện ảnh trùng / gần trùng / mâu thuẫn nhãn |
| `inspect_data.py` | Kiểm tra dữ liệu, sinh `images.csv`, `near_duplicate_pairs.csv`, `samples.png` |
| `visualize_augmentation.py` | Sinh hình minh hoạ augmentation cho báo cáo |
| `requirements.txt` | Thư viện cần cài |

---

## 13. Trạng thái kiểm thử và việc cần làm trước khi nộp

**Đã kiểm tra:**
- Logic quét ảnh, chia stratified, chia theo nhóm, chống trùng: chạy trên dữ liệu giả lập có cài sẵn trường hợp trùng y hệt, trùng khác kích thước, xung đột nhãn, file hỏng (đều xử lý đúng).
- Mô phỏng toàn bộ logic làm sạch và chia tập trên `images.csv` thật: ra đúng số liệu ở mục 4 và 5.

**Chưa kiểm tra bằng dữ liệu thật** (môi trường soạn code không có torch và không tải được dataset):
- [ ] Chạy `python dataset.py`: xác nhận in ra Train/Val/Test ≈ 776/97/98, shape batch `(32, 3, 224, 224)`, và dòng "Train has augmentation: True / Val has augmentation: False".
- [ ] Chạy `python visualize_augmentation.py`, xem 2 hình có hợp lý không (xoay 15° và vùng đen khi zoom nhỏ có làm hỏng ảnh không), rồi chèn vào mục 6.
- [ ] Mở thử 3–4 cặp trong `inspect_output/near_duplicate_pairs.csv` xem có đúng là cùng một hình không.
- [ ] Điền kết quả thật nếu số liệu lệch so với mục 4, 5.
