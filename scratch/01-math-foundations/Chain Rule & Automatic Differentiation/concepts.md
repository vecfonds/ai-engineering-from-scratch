The chain rule is the engine behind every neural network that learns.
Quy tắc dây chuyền là động cơ đằng sau mọi mạng lưới thần kinh học hỏi.

Linear Algebra giúp bạn hiểu dữ liệu được biểu diễn như thế nào.
Calculus giúp bạn hiểu đạo hàm và Gradient.
Bài này sẽ giải thích cách các Neural Network tính toán Gradient để tự học.

1. Chain Rule - Quy tắc dây chuyền
1.1 Tại sao cần Chain Rule?
Khi y không phụ thuộc trực tiếp vào x thông qua một phép tính duy nhất. Nó phụ thuộc vào kết quả của một phép tính trung gian.
=> sử dụng Chain Rule

1.2 Công thức Chain Rule
Muốn biết ảnh hưởng của x đến y, ta nhân mức độ ảnh hưởng của từng bước trung gian với nhau.

Mỗi bước có một đạo hàm cục bộ (Local Derivative). Chain Rule kết hợp các đạo hàm cục bộ để tìm đạo hàm tổng thể.

2. Computational Graph - Đồ thị tính toán
Chain Rule cho chúng ta công thức toán học.
Nhưng khi Neural Network có hàng triệu tham số và hàng nghìn phép tính, việc tính đạo hàm thủ công sẽ rất khó.
Để giải quyết vấn đề này ta dùng Computational Graph.

2.1 Computational Graph là gì?
Computational Graph là đồ thị biểu diễn các phép tính và mối quan hệ phụ thuộc giữa chúng.

Mỗi node đại diện cho một giá trị hoặc một phép toán. Các cạnh thể hiện mối quan hệ phụ thuộc giữa các giá trị.

2.2 Forward Pass - Lan truyền xuôi
Forward Pass là quá trình tính toán từ Input đến Output.

VD: y = (x1 * x2) + b
Với x1=2, x2=3, b=1

a = x1 * x2 = 2*3 = 6

y = a+b = 6+1 = 7

Kết quả: y=7

Đây chính là quá trình Neural Network nhận dữ liệu đầu vào và đưa ra dự đoán.



2.3 Backward Pass - Lan truyền ngược
Giả sử chúng ta myoons biết:
- Nếu thay đổi x1, kết quả y thay đổi thế nào?
- Nếu thay đổi x2, kết quả y thay đổi thế nào?
- Nếu thay đổi b, kết quả y thay đổi thế nào?

Chúng ta cần tính đạo hàm riêng: dy     dy   dy
                                 --  ,  -- , -- 
                                 dx1    dx2  db

3. Forward Mode vs Reverse Mode
Có 2 phương pháp phổ biến để thực hiện Automatic Differentiation.

3.1 Forward Mode
Forward Mode truyền đạo hàm từ Input đến Output.
VD:
        x=2
        a=x^2
        y=sin(a)

Ta bắt đầu với:
        dx
        --=1
        dx
Tiếp tục:
        da
        --=2x=4
        dx
Cuối cùng:
        dy          da
        -- = cos(a)* --
        dx          dx
           = cos(4)*4
           = -2.615

3.2 Reverse Mode
Reverse Mode thực hiện theo chiều ngược lại, bắt đầu từ Output.
        y= sin(a)
        a=x^2

Bắt đầu:
        dy
        --=1
        dy
Tính ngược:
        dy
        -- = cos(a) = cos(4)
        da

Tiếp tục:
        dy   dy   da
        -- = -- x --
        dx   da   dx
           = cos(4) * 4
           = -2.615

Hai phương pháp cho cùng kết quả.

3.3 So sánh 2 phương pháp

Tiêu chí        Forward Mode                        Reverse Mode
Hướng tính      Input -> Output                     Output -> Input
Khởi tạo        Input gradient = 1                  Output gradient = 1
Phù hợp         ít input, nhiều output              nhiều input, ít output
Ứng dụng        Một số bài toán đạo hàm             Neural Network

Tại sao Neural Network sử dụng Reverse Mode?
Giả sử một mô hình có triệu weights nhưng chỉ tạo ra một gía trị Loss.
- Forward Mode: về cơ bản cần một lượt lan truyền đạo hàm cho mỗi input độc lập.
- Reverse Mode: có thể tính gradient đối với tất cả weight trong một lượt Backward Pass.
Vì vậy, Reverse Mode rất phù hợp với việc huấn luyện Deep Learning.




- Forward Pass và Backward Pass: hai giai đoạn tính toán trong quá trình huấn luyện Neural Network.
    Forward Pass: Input -> Model -> Prediction -> Loss
    Backward Pass: Loss -> Gradients -> Model Parameters

- Forward Mode và Reverse Mode: Hai phương pháp tính đạo hàm trong Automatic Differentiation.
    Forward Mode: Đạo hàm truyền xuôi cùng quá trình tính toán
    Reverse Mode: Đạo hàm truyền ngượ từ output về input

1. Forward Pass - Lan truyền xuôi
Forward Pass là qúa trình đưa dữ liệu đầu vào qua các phép tính của mô hình để tạo ra kết quả dự đoán.

