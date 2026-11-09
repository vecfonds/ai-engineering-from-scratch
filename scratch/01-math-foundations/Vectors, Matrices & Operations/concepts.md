Every neural network is just matrix multiplication with extra steps.

Every image your model processes is a matrix of pixel values. 
Mỗi hình ảnh mà mô hình của bạn xử lý là một ma trận các giá trị pixel.

Every work embedding is a vector.
Mỗi từ núng là một vector.

Every layer of every neural network is a matrix transformation.
Mỗi lớp của mỗi mạng lứoi thần kinh là một phép biến đổi ma trận.

You cannot build AI systems without being fluent in matrix operations the same way you cannot write code without understanding variables.
Bạn không thể xây dựng hệ thống AI nếu không thông thạo các phép toán ma trận giống như cách bạn không thể viết mã mà không hiểu các biến.


1. Vector và Matrix Operations
1.1 Vector là gì?
Vector là một danh sách các số có hướng và độ lớn.
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

Trong Neural Networks, ma trận trọng số biến đổi vector đầu vào thành vector đầu ra. Một layer có 784 input và 128 output sử dụng ma trận trọng số 128x784.

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
- Điều này có nghĩa phép biến đổi đã làm mất ít nhất một chiều của không gian. Trong 2D, một vùng có diện tích dương có thể bị ép thành một đường thẳng hoặc một điểm.
- Ma trận không có nghịch đảo
- Hệ tuyến tính sử dụng nó không có nghiệm hoặc có vô số nghiệm

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

6. Identity Matrix - Ma trận đơn vị
6.1 Identity Matrix là gì?
Identity Matrix (ma trận đơn vị) là một ma trận vuông có:
- các phần tử trên đường chéo chính bằng 1.
- tất cả phần tử còn lại bằng 0
Bạn có thể hình dung Identity Matrix giống như số 1 trong phép nhân số học.
Với mọi ma trận A phù hợp kích thước:
        A @ I = I @ A = A
Ma trận A không thay đổi.

6.2 Identity Matrix liên quan gì đến Inverse?
Đây là mối quan hệ quan trọng nhất:
        A @ A-1 = I
Trong đó:
- A: phép biến đổi ban đầu
- A-1: phép biến đổi ngược
- I: kết quả khi hai phép biến đổi triệt tiêu nhau

6.3 Mối liên hệ giữa 3 khái niệm
Khái niệm       Ý nghĩa                                   Điều cần nhớ
Determinant     Hệ số biến đổi diện tích/thể tích có dấu  det=0 thì không khả nghịch
Iverse          đảo ngược phép biến đổi                   chỉ tồn tại khi det!=0
Identity        phép biến đổi không làm thay đổi dữ liệu  A @ I = A

Mối quan hệ:
                det(A) != 0 <=> A-1 tồn tại
với ma trận vuông.

7. Broadcasting

7.1 Broadcasting là gì?
Broadcasting là cơ chế cho phép Numpy thực hiện các phép toán giữa những array có shape khác nhau bằng cách tự động xem chúng như được mở rộng theo các chiều phù hợp, mà không nhất thiết phải tạo ra bản sao dữ liệu thực sự.

7.2 Quy tắc Broadcasting
Numpy so sánh Shape của 2 array từ chiều cuối cùng trở về chiều đầu tiên (Right to Left)
Hai chiều được xem là tương thích nếu:
- Hai kích thước bằng nhau
- Một trong hai kích thước bằng 1
- Một array không có chiều tương ứng thì được xem như có kích thước 1 ở phía trước.
Nếu không đáp ứng đủ các điều kiện trên thì Broadcasting sẽ thất bại.

VD1: Broadcasting hợp lệ
        A:(2,3)
        B:(3,)
Ta có thể hình dung B được mở rộng thành
        B:(1,3) -> (2,3)

VD2: Broadcasting hợp lệ
        A:(3,1)
        B:(1,4)
So sánh từng chiều:
        3 x 1
        1 x 4
Kết qủa: A+B = (3:4)

7.3 Broadcasting trong Neural Network: Bias Addition
Giả sử một Neural Network có 3 neurons và nhận dữ liệu từ 2 samples.
Sau khi nhân Weight Matrix với Input Matrix,  chúng ta nhận được:
        Z = 1 2 3
            4 5 6 
Trong đó:
- 2 hàng: 2 samples
- 3 cột: 3 neurons

Bias của layer: b =[10,20,30]
Ta muốn cộng bias vào từng sample
        Z + b = 1+10    2+20    3+30
                4+10    5+20    6+30
             
              = 11 22 33
                14 25 36
Điều quan trọng: Bias của mỗi neuron được cộng vào tất cả samples, chứ không phải mỗi sample có một bộ bias riêng.

8. Neural Network Forward Pass
8.1 Forward Pass là gì?
Forward Pass (lan truyền thuận) là quá trình dữ liệu đi từ đầu vào (Input) qua các layer của Neural Network để tạo ra kết quả dự đoán (Output).
Một Dense Layer cơ bản thực hiện 3 bước:
- Matrix Multiplication
- Bias Addition
- Activation Function

Công thức:      y = ReLU(W@x+b)
Trong đó:
        x: Input vector
        W: Weight matrix
        b: Bias vector
        W@x: Weighted Sum
        +b: Bias Addition
        ReLU: Activation Function
        y: Output vector

8.1 Bước 1 - Input layer
Giả sử chúng ta có một căn nhà với 3 đặc trưng:
                0.5
        x =     0.8
                0.2
Các giá trị này đã được chuẩn hoá (Normalized).
Shape:  x:(3,1)

Có 3 input features.

8.3 Bước 2 - Weight Matrix
Giả sử Hidden Layer có 2 Neurons.
Mỗi Neuron cần 3 weights để kết nối với 3 input features.

Do đó:          W =     0.2 0.4 0.6
                        0.1 0.3 0.5

Shape:  W:(2,3)
- 2 hàng: 2 neurons
- 3 cột: 3 weights cho mỗi neuron.

8.4 Bước 3 - matrix multiplication
Chúng ta thực hiện:     Z=W@x
                         = 0.2 0.4 0.6 | 0.5
                           0.1 0.3 0.5 | 0.8
                                       | 0.2

                         = 0.54
                           0.39

Shape:  (2,3)@(3,1)=(2,1)

8.5 Bước 4 - Bias Addition
Giả sử: b = 0.1
            -0.2

Ta có Z+b = 0.54        0.1
            0.39        -0.2
          
          = 0.64
            0.19

Kết quả này được gọi là Pre-activation (giá trị trước kích hoạt)

8.6 Bước 5 - Activation Function: ReLU
ReLU là viết tắt của Rectified Linear Unit

Công thức:      ReLU(x) = max(0,x)

Nghĩa là:
- Nếu x lớn hơn 0: giữ nguyên
- Nếu x nhỏ hơn hoặc bằng 0: trả về 0

VD: ReLU(-2)=0
    ReLU(3)=3

Áp dụng vào kết quả của chúng ta:
        ReLU(0.64) = 0.64
             0.19    0.19

Trong NumPy:
output = np.maximum(0, W@x + b)


9. Tổng kết
- Vector là một danh sách có số có thứ tự. Trong AI: một điểm trong không gian nhiều chiều.
- Matrix: Một phép biến đổi tuyến tính. Nó ánh xạ các vector từ không gian này sang không gian khác.
- Matrix multiply: tích vô hướng giữa mỗi hàng của ma trận đầu tiên và mỗi cột của ma trận thứ 2. Thứ tự quan trọng.
- Transpose: hoán đổi hàng và cột. Biến một ma trận mxn thành nxm. Quan trọng trong lan truyền ngược (backpropagation).
- Determinant: đo diện tích quy mô ma trận (2D) hoặc thể tích (3D). Số 0 có nghĩa là sự biến đổi sẽ nghiền nát một chiều.
- Inverse: ma trận đảo ngược phép biến đổi. Chỉ tồn tại khi định thứ khác 0.
- Identity matrix: ma trận tương đường với phép nhân với 1. Được sử dụng trong các kết nói dư (ResNets).
- Broadcasting: kéo dài mảng nhỏ hơn để khớp với mảng lớn hơn bằng cách lặp lại dọc theo các thứ nguyên bị thiếu.
- Element-wise: nhân các vị trí phù hợp. Cả 2 mảng phải có hình dạng giống nhau (hoặc có thể broadcast).


Tóm tắt các phép toán ma trận và ý nghĩa của chúng trong neural network:
- Addition (công ma trận): cộng từng phần tử tương ứng.
        Trong mạng neuron: Adding bias to output. Mỗi lớp tính y=Wx+b, trong đó b dịch chuyển kết quả để mô hình linh hoạt hơn (không bị ép đi qua gốc toạ độ).

- Scalar Multiply (nhân với một số): nhân mọi phần tử với cùng một số.
        Trong mạng nơ-ron: learning rate × gradients. Khi cập nhật trọng số: W ← W − η·∇W, thì η chính là số vô hướng làm co giãn bước đi.

- Matrix Multiply (nhân ma trận): biến đổi vector 
        Trong mạng nơ-ron: Layer forward pass. Phép Wx chính là lớp fully-connected, và đây là phép tốn nhiều tính toán nhất (nên GPU được tối ưu cho nó).

- Transpose (chuyển vị): đổi hàng thành cột.
        Trong mạng nơ-ron: backpropagation. Gradient lan truyền ngược dùng Wᵀ, ví dụ ∂L/∂x = Wᵀ · ∂L/∂y.

- Determinant (định thức): một con số tóm tắt ma trận (độ co giãn diện tích/thể tích; bằng 0 nghĩa là ma trận làm “xẹp” một chiều).
        Trong mạng nơ-ron: kiểm tra khả nghịch. det ≠ 0 thì ma trận khả nghịch; det = 0 thì suy biến. Liên hệ với bảng đầu tiên: rank deficient ⇒ det = 0.

- Inverse (nghịch đảo): hoàn tác một phép biến đổi (A⁻¹A = I).
        Trong mạng nơ-ron: giải hệ tuyến tính, ví dụ phương trình chuẩn của linear regression w = (XᵀX)⁻¹Xᵀy. Thực tế người ta thường không tính nghịch đảo trực tiếp mà dùng QR hoặc SVD cho ổn định.

- Identity (ma trận đơn vị): ma trận “không làm gì”, Ix = x.
        Trong mạng nơ-ron:
        - Khởi tạo: đôi khi khởi tạo trọng số gần I để lớp ban đầu giữ nguyên tín hiệu.
        - Residual connection (ResNet, Transformer): y = x + F(x), phần “x” chính là đường đi qua phép đồng nhất, giúp gradient truyền qua mạng sâu dễ hơn.

Ví dụ gom lại một lớp mạng:
        Một lớp: y = Wx + b
- Wx: matrix multiply
- b: addition
- Backprop: dùng Wᵀ
- Cập nhật: W ← W − η·∇W (scalar multiply)