Every neural network is just matrix multiplication with extra steps.

1. Vector và Matrix Operations
1.1 Vector là gì?
Vector là một dãy số có thứ tự.
Trong AI, một vector thường đại diện cho một đối tượng thông qua các đặc trưng (features).

Ví dụ: một căn nhà thường được biểu diễn bằng 3 features:
    x = [100 3 2]
trong đó: 
- 100: diện tích (m2)
- 3: số phòng ngủ
- 2: số phòng tắm

Mỗi phần tử trong vector mang một ý nghĩa cụ thể.

Lưu ý: Vector trong Machine Learning có thể có hàng trăm, hàng nghìn hoặc hàng triệu chiều. Không nhất thiết chỉ có 2D hoặc 3D như trong hình học thông thường.

1.2 Matrix là gì?
Matrix (ma trận) là một bảng gồm nhiều hàng (rows) và nhiều cột (columns)

VD: 
3 hàng = 3 samples (3 căn nhà); 3 cột = 3 features.
Nếu mỗi hàng đại diện cho một căn nhà, thì ma trận chứa dữ liệu của 3 căn nhà.

Đây chính là cách dữ liệu thường được tổ chức trong Machine Learning:
- Sample: một mẫu dữ liệu
- Feature: một đặng trưng
- Dataset: tập hợp nhiều mẫu dữ liệu

1.3 Shape - kích thước của Vector và Matrix
Shape cho biết số lượng phần tử theo từng chiều.

VD: 1 2 3
    4 5 6

Ta có: Shape(A) = (2,3)
Tức là: 
- Dimension 1: 2 rows
- Dimension 2: 3 columns

Một số thuật ngữ cần nhớ:
Shape       kích thước theo từng chiều
ndim        số chiều
size        tổng số phần tử 
axis        trục của mảng

1.4 Các phép toán cơ bản
Chúng ta cần phân biệt 3 phép toán:
- addition: cộng 2 ma trận
- subtraction: trừ 2 ma trận
- scalar multiplication: nhân ma trận với một số vô hướng

Điểm quan trong: Các phép cộng và trừ thông thường yêu cầu 2 ma trận có cùng Shape
Trong AI, các phép toán này được dùng để xử lý dữ liệu, cập nhật tham số và tính toán gradient.

2. Element-wise Operations vs Matrix Multiplication
2.1 Element-wise Multiplication là gì?
Element-wise Multiplication (phép nhân từng phần tử) nghĩa là lấy từng phần tử của ma trận A nhân với phần tử tưởng ứng cùng vị trí trong ma trận B.

Điều kiện: Hai ma trận phải cùng Shape.

Trong Numpy, toán tử * thực hiện Element-wise Multiplication đói với 2 array tương thích.

2.2 Matrix Multiplication là gì?
Mỗi phần tử của ma trận kết quả được tính bằng tích vô hướng (Dot Product) giữa một hàng của A và một cột của B.

Trong Numpy, sử dụng toán tử @.

3. Transpose - Chuyển vị ma trận

3.1 Transpose là gì?
Transpose (chuyển vị) là thao tác đổi hàng thành cột và đổi cột thành hàng.

3.2 Tại sao Transpose quan trọng trong AI?
Transpose có thể giúp thay đổi Shape để phép nhân ma trận trở nên hợp lệ.
Trong Neural Network, Transpose còn xuất hiện trong Backpropagation (lan truyền ngược), đặc biệt khi tính gradient của weights và inputs.

Một lưu ý quan trọng: Với NumPy, nếu x là array một chiều có shape (3,), thì x.T vẫn có shape (3,). Muốn chuyển vector hàng thành vector cột, bạn cần x.reshape(3, 1).

4. Determinant - Định thức

4.1 Determinant là gì?
là một giá trị số được tính từ một ma trận vuông.
Ký hiệu: det(A) hoặc |A|

4.2 Ý nghĩa hình học của Determinant
Một ma trận có thể biểu diễn một phép biến đổi không gian (Linear Transformation)

VD: một ma trận 2x2 biến đổi các vector trong mặt phẳng 2D.
Giả sử ban đầu chúng ta có một hình vuông với diện tích bằng 1.
Sau khi áp dụng một phép biến đổi tuyến tính, hình vuông có thể:
- phóng to hoặc thu nhỏ
- xoay
- kéo nghiên thành hình bình hàng
- bị ép thành một đường thẳng

Determinant cho biết hệ số thay đổi diện tích có xét đến dấu và hướng.

VD:
Determinant     Ý nghĩa hình học
2               diện tích tăng gấp đôi
0.5             diện tích giảm còn một nửa
1               diện tích không đổi
-1              diện tích không đổi nhưng đảo hướng
0               không gian bị ép xuống chiều thấp hơn

Điểm quan trọng nhất: det(A) = 0
Điều này có nghĩa phép biến đổi đã làm mất ít nhất một chiều của không gian. Trong 2D, một vùng có diện tích dương có thể bị ép thành một đường thẳng hoặc một điểm.

5. Inverse Matrix - Ma trận nghịch đảo
5.1 Inverse Matrix là gì?
Inverse matrix là ma trận có khả năng đảo ngược phép biến đổi do ma trận ban đầu tạo ra.
VD: nếu A biến đổi một vector x thành y:
        y = A @ x
Thì nếu A có ma trận nghịch đảo:
        x = A-1 @ y

Chúng ta có thể khôi phục lại vector ban đầu.

Điều kiện: det(A) != 0
Nếu Determinant bằng 0 thì ma trận không có nghịch đảo.

Cho ma trận A và ma trận nghịch đảo của nó là A-1.
Ta có:      A @ A-1 = I
Trong đó: I là Identity Matrix (ma trận đơn vị)

Trong Numpy:
det_A = np.linalg.det(A)
A_inv = np.linalg.inv(A)
result = A @ A_inv

Lưu ý: np.linalg.inv() sẽ báo LinAlgError nếu ma trận suy biến (singular), tức là không có nghịch đảo.

5.2 Ứng dụng của Inverse Matrix trong AI.
Inverse Matrix xuất hiện trong nhiều lĩnh vực:
- Linear Algebra: Giải hệ phương trình tuyến tính.
- Linear Regression: Nghiệm dạng đóng (Normal Equation).
- Optimization: Một số phương pháp tối ưu hóa.
- Computer Vision: Các phép biến đổi hình học.