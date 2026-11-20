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
Giả sử chúng ta muốn biết:
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
    Reverse Mode: Đạo hàm truyền ngược từ output về input

Bạn hãy ghi nhớ một ý cốt lõi:

Forward Pass không chỉ để tạo ra Prediction và Loss. Nó còn chuẩn bị và lưu những giá trị cần thiết để Reverse Mode tính Gradient ở lượt Backward Pass.

Đó chính là lý do hai giai đoạn này phải phối hợp với nhau.

4. Xây dựng Autograd Engine bằng Python

4.1 Autograd là gì?
Autograd viết tắt của Automatic Differentiation (vi phân tự động).
Đây là cơ chế tự động tính đạo hàm của một hàm số thông qua các phép toán đã thực hiện.

PyTorch không cần bạn viết công thức đạo hàm. Nó theo dõi các phép toán, xây dựng Computational Graph và tính Gradient tự động.

4.2 Thiết kế một Autograd Engine mini
Một Node trong Computational Graph cần có:
        data            giá trị của node / giá trị Forward
        grad            Gradient của node
        _prev           Các node đầu vào
        _backward       Hàm tính Gradient ngược
        op              phép toán tạo ra node

4.3 Xây dựng lớp Value

Class Value:
        def __init__(self, data, _children=(), op=''):
                self.data = data
                self.grad = 0.0

                self._prev = set(_children)
                self._op = op

                self._backward = lambda: None
        
        def __repr__(self):
                return f"Value(data={self.data}, grad={self.grad})"

- data: lưu giá trị số của node.
- grad: lưu đạo hàm của node đối với node đầu ra cuối cùng (sau khi thực hiện Backward Pass giá trị này sẽ được cập nhật).
- _prev: lưu những node tạo ra node hiện tại (tham chiếu).
- _backward: chứa logic để truyền Gradient ngược về các node trước đó. (ban đầu ta chưa thực hiện phép toán nào nên sử dụng hàm rỗng)

4.4 Xây dựng phép công - Addition
        def __add__(self, other):
                other = other if isinstance(other, Value) else Value(other)

                out = Value(
                        self.data + other.data,
                        (self, other),
                        '+'
                )

                def _backward():
                        self.grad += out.grad
                        other.grad += out.grad
                
                out._backward = _backward

                return out

4.5 Xây dựng phép nhân - Multiplication

        def __mul__(self, other):
                other = other if isinstance(other, Value) else Value(other)

                out = Value(
                        self.data*other.data,
                        (self,other),
                        '*'
                )

                def _backward():
                        self.grad += other.data*out.grad
                        other.grad += self.data*out.grad

                out._backward = _backward
                return out

4.6 Thêm các phép toán khác
Để có thể sử dụng thuận tiện hơn, chúng ta bổ sung phép lũy thừa, phép chia và hỗ trợ số ở bên trái.

5. Backward Pass - Tự động truyền Gradient

5.1 Topological Sort là gì?
Giả sử có đồ thị:       x -> a -> b -> y

Khi truyền Gradient ngược, chúng ta phải bắt đầu từ y sau đó đến b, a và cuối cùng là x.
Nếu gọi _backward() tuỳ tiện, một node có thể được tính Gradient trước khi nhận đầy đủ Gradient từ các node phía sau.

Topological Sort giúp xác định thứ tự xử lý hợp lệ của các node trong Computational Graph.

5.2 Xây dựng hàm backward()

        def backward(self):
                topo = []
                visited = set()

                def build_topo(v):
                        if v not in visited:
                                visited.add(v)

                                for child in v._prev:
                                        build_topo(child)
                                
                                topo.append(v)
                        
                build_topo(self)
                
                self.grad = 1

                for v in reversed(topo):
                        v._backward()

6. Liên hệ với Neural Network
Một Neural Network có các thành phần:
- Input: nhận dữ liệu
- Weights: trong số của các kết nối
- Bias: đọ lệch
- Activation Function: hàm kích hoạt
- Output: kết quả dự đoán
- Loss Function: đo sai số dự đoán

Quy trình huấn luyện:
        Input 
        -> Forward Pass (tính toán Prediction)
        -> Loss Function (tính sai số)
        -> Backward Pass (tính gradient bằng autograd)
        -> Optimizer (cập nhật các Weights và Bias)
Lặp lại cho đến khi mô hình học được.
=> đây chính là nền tảng của quá trình Training trong Deep Learning.


