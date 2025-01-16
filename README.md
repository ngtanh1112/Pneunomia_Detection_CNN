# Pneumonia Detection Using CNN

## 1. Giới thiệu

Viêm phổi là một trong những bệnh hô hấp nguy hiểm nhất, đặc biệt ở trẻ nhỏ. Việc phát hiện sớm viêm phổi dựa trên ảnh X-quang ngực là một công việc phức tạp đòi hỏi kinh nghiệm từ các chuyên gia. Trong dự án này, sỬ dụng mô hình Mạng Nơ-ron Tích chập (CNN) để phát triển hệ thống nhận diện viêm phổi tự động dựa trên ảnh chụp X-quang.

## 2. Lấy dữ liệu

Bộ dữ liệu được sử dụng trong dự án này là bộ **Chest X-Ray Images (Pneumonia)** được cung cấp trên Kaggle:

[Kaggle: Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

### Mô tả bộ dữ liệu ảnh viêm phổi

- **Bố cục**: Bộ dữ liệu bao gồm 3 thư mục chính:

  - `train`: Chứa dữ liệu huấn luyện.
  - `test`: Chứa dữ liệu kiểm tra.
  - `val`: Chứa dữ liệu xác thực.

- **Phân loại**: Mỗi thư mục chính lại chứa hai thư mục con:

  - `PNEUMONIA`: Chứa ảnh X-quang bị viêm phổi.
  - `NORMAL`: Chứa ảnh X-quang bình thường.

- **Số lượng dữ liệu**: Tổng cộng 5.863 ảnh X-quang, bao gồm 2 nhóm chính:

  - `PNEUMONIA`.
  - `NORMAL`.

- **Nguồn dữ liệu**: Ảnh X-quang ngực được thu thập từ các bệnh nhân nhi trong độ tuổi từ 1 đến 5 tuổi.

## 3. Cách thay đổi đường dẫn dữ liệu trong code

Trước khi bắt đầu, cần chắc chắn đường dẫn dữ liệu trong mã nguồn khớp với địa chỉ lưu trữ của máy.

### Mã minh họạ:

```python
import numpy as np
import pandas as pd
import os

# Kiểm tra danh sách các file trong thư mục dữ liệu
for dirname, _, filenames in os.walk('D:/HUS_22001541/ComputerVision/FinalReport/source/input/chest_xray'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Đổi đường dẫn dữ liệu trong code
train = get_training_data('D:/HUS_22001541/ComputerVision/FinalReport/source/input/chest_xray/train')
test = get_training_data('D:/HUS_22001541/ComputerVision/FinalReport/source/input/chest_xray/test')
val = get_training_data('D:/HUS_22001541/ComputerVision/FinalReport/source/input/chest_xray/val')
```

- **Lưu ý**: Thay thế đường dẫn `D:/HUS_22001541/ComputerVision/FinalReport/source/input/chest_xray` bằng đường dẫn đến thư mục dữ liệu của bạn.

## 4. Quy trình huấn luyện mô hình

### 4.1 Xử lý dữ liệu

- Đọc ảnh từ bộ dữ liệu.
- Chuẩn hóa dữ liệu (đổi kích thước ảnh về 150x150 pixel, chuẩn hóa giá trị pixel trong khoảng [0,1]).
- Phân chia dữ liệu thành tập huấn luyện, tập kiểm tra và tập xác thực.

### 4.2 Xây dựng mô hình CNN

- Kiến trúc CNN gồm các lớp tích chập, lớp pooling, lớp dàn phẳng (flatten), và các lớp fully-connected.
- Sử dụng hàm loss: `binary_crossentropy`.
- Số epoch: 12 (có thể thay đổi).

### 4.3 Đánh giá mô hình

- Accuracy: ~90%
- Loss: ~0.2

## 5. Yêu cầu hệ thống

- RAM > 8GB (đề xuất 16GB).
- GPU hỗ trợ CUDA (đề xuất sử dụng Google Colab Pro).

## 6. Tài liệu tham khảo

- [Kaggle Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- Tài liệu về Mạng Nơ-ron Tích chập (CNN).

