# IrisApp
# 🌸 IrisApp - Dự đoán loài hoa Iris bằng KNN và Flask

Ứng dụng web đơn giản sử dụng **Flask** và **K-Nearest Neighbors (KNN)** để dự đoán loài hoa Iris dựa trên 4 tham số đầu vào:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

## 🚀 Tính năng
- Giao diện web thân thiện (HTML + Flask)
- Người dùng nhập 4 tham số của hoa
- Trả về loài hoa tương ứng (Setosa, Versicolor, Virginica)
- Mô hình KNN huấn luyện trực tiếp từ dataset **Iris.csv**

## 🛠 Cài đặt

### 1. Clone repo
```bash
## git clone https://github.com/BaNguyen0611/IrisApp.git
cd IrisApp
2. Cài dependencies
pip install -r requirements.txt

3. Chạy ứng dụng
python app.py


Ứng dụng sẽ chạy tại:
👉 http://127.0.0.1:5000

📂 Cấu trúc dự án
IrisApp/
 ├── app.py              # Flask app chính
 ├── requirements.txt    # Danh sách dependencies
 ├── Iris.csv            # Dataset huấn luyện
 └── templates/
     └── index.html      # Giao diện web
