Training a neural network is nothing more than finding the bottom of a valley.
Huấn luyện 1 mạng lưới thần kinh không gì khác hơn là tìm đáy thung lũng.

1. Opimization là gì?
Optimization (tối ưu hóa) là quá trình tìm kiếm các giá trị đầu vào để làm cho 1 hàm số đạt giá trị nhỏ nhất hoặc lớn nhất.
Trong ML, chúng ta muốn tìm các Weights và Biases sao cho Loss Function nhỏ nhất.

1.2 Optimization liên quan gì đến Neural Network?
Một Neural Network có thể chứa hàng triệu hoặc hàng tỷ Parameters.
VD, mô hình có:
    W = {w1, w2, ... , wn}
Mỗi weight ảnh hưởng đến kết quả dự đoán của mô hình.

Quá trình Training về cơ bản:
    Đưa dữ liệu vào Neural Network
    Mô hình tạo ra dự đoán
    Tính Loss để đo mức độ sai lệch
    Tính Gradient để biết càn điều chỉnh Weight theo hướng nào
    Optimization Algorithm cập nhật Weight
    Lặp lại quá trình trên

Điểm quan trọng:
    Loss Funciton: cho biết mô hình sai bao nhiêu
    Gradient: cho biết Loss thay đổi theo từng Weight như thế nào
    Optimizer: quyết định cách cập nhật weight

2. Gradient Descent
- Gradient cho biết hướng và tốc độ thay đổi của Loss theo Parameters
- Gradient Descent di chuyển ngược Gradient để giảm Loss
- Parameters được cập nhật lặp đi lặp lại theo công thức
    W(new) = W(old)-a dentaL(W)

3. Learning Rate - Tốc độ học

3.1 Learning Rate là gì?
Learning Rate (LR) là một Hyperparameter quyết định kích thước bước cập nhật Parameters trong mỗi lần thực hiện Gradient Descent.
Nói đơn giản:
    Gradient quyết định hướng đi
    Learning Rate quyết định khoảng cách di chuyển

3.2 Learning Rate quá nhỏ (too small)
- Training rất chậm
- Cần nhiều Epochs hơn
- Tốn thời gian và tài nguyên tính toán
- Có thể chưa đạt được nghiệm tốt trong giới hạn thời gian Training

3.3 Learning Rate quá lớn (too large)
Nếu LR quá lớn, mô hình có thể:
- Liên tục vượt qua điểm tối ưu
- Dao động quanh điểm tối ưu
- Loss tăng giảm thất thường
- Thậm chí Diverge (phân kỳ), khiến Loss ngày càng lớn

3.4 Learning Rate phù hợp (Good Learing Rate)
Không tồn tại một LR tối ưu cho tất cả mô hình và Dataset.
LR phù hợp phụ thuộc vào:
    Dataset
    Model Architecture
    Loss Function
    Optimizer
    Training Configuration

