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

6. Backpropagation trong Neural Network
Làm thế nào để Neural Network biết phải thay đổi từng Weight và Bias bao nhiêu để dự đoán chính xác hơn?
-> Backpropagation kết hợp với Gradient Descent.

6.1 Cấu trúc của một Neural Network

Một Neural Network có các thành phần:
- Input Layer: nhận dữ liệu đầu vào
- Hidden Layer: học các đặc trưng từ dữ liệu
- Output Layer: tạo ra dự đoán
- Weights: trong số của các kết nối, điều chỉnh mức độ ảnh hưởng giữa các Neuron.
- Bias: độ lệch, điều chỉnh giá trị đầu ra của Neuron.
- Activation Function: hàm kích hoạt, tạo tính phi tuyến cho mô hình.
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

Neural Network sẽ học các Weights và Biases để đưa ra dự đoán phù hợp với dữ liệu thực tế.


6.2 Một Neuron hoạt động như thế nào?
Đây là thành phần cơ bản nhất của Neural Network.
Một Neuron nhận nhiều đầu vào, nhân từng đầu vào với Weight tương ứng, cộng Bias rồi đưa qua Activation Function.

Công thức:      z = w1x1 + w2x2 + b

Sau đó:         a = f(z)

Trong đó:
- x1, x2: input
- w1, w2: Weights
- b: Bias
- z: Weighted Sum
- f: Activation Function
- a: Output của Neuron

Một Artificial Neuron thường có 4 thành phần:
- Inputs (đầu vào)
        Giả sử chúng ta muốn xây dựng một mô hình dự đoán giá nhà dựa trên 2 yếu tố:
        x1: diện tích căn nhà
        x2: số phòng ngủ

        VD: x1=100      x2=3
        Neuron nhận 2 giá trị này làm đầu vào

- Weights (trong số)
        Mỗi input sẽ có một trong số tương ứng:
        w1: mức độ ảnh hưởng của diện tích
        w2: mức độ ảnh hưởng của số phòng ngủ

        Giả sử: w1=0.8  w2=0.5
        Trọng số thể hiện mức độ quan trọng mà mô hình gán cho từng input.

        Neuron thực hiện phép nhân từng input với trọng số tương ứng:
                x1w1 + x2w2
        Thay số:
                100*0.8+3*0.5=81.5
        
- Bias (độ lệch)
        Bias là một tham số bổ sung được cộng vào tổng có trọng số.

        Giả sử: b=2
        khi đó:
                z = x1w1 + x2w2 + b
                z = 100*0.8 + 3*0.5 + 2 = 83.5
        
        Bias giúp mô hình linh hoạt hơn trong việc học mối quan hệ giữa input và output.

-Activation Function (hàm kích hoạt)
        Sau khi tính toán z, neuron đưa giá trị này qua một Activation Function.
                a = f(z)
        Trong đó:
                z: tổng có trọng số cộng bias
                f: Activation Function
                a: output của neuron
        
        Một số Activation Function phổ biến:
        - ReLU: f(x) = max(0,x)
                Giá trị âm thành 0, giá trị dương được giữ nguyên.
        - Sigmoid
                Đưa giá trị về khoảng (0,1)
        -Tanh:  f(x) = tanh(x)
                Đưa giá trị về khoảng (-1,1)
        
6.3 Tại sao cần Activaton Function?
Nếu không có Activation Function phi tuyến, việc xếp chồng nhiều Linear Layer vẫn chỉ tạo ra một phép biến đổi tuyến tính.

        a = w1x1 + w2x2 + b
VD:
        h = W1x + b1
        y = W2h + b2

Thay h vào:
        y = W2(W1x+b1) + b2
        y = (W2W1)x + (W2b1 + b2)

Đây vẫn là một phép biến đổi tuyến tính.
Activation Function giúp Neural Network học được các quan hệ phi tuyến phức tạp.

Một số Activation Function phổ biến:
- ReLU: được sử dụng rộng rãi trong Hidden Layers
- Sigmoid: đầu ra nằm trong khoảng (0,1), thường dùng ở Output Layer cho binary classification
- Tanh: đầu ra nằm trong khoảng (-1,1)

Một điểm rất quan trọng: Activation Function cũng là một phép toán trong Computational Graph. Vì vậy, khi Backward Pass diễn ra, chúng ta phải tính đạo hàm của nó.

VD:                | 1 x>0
        ReLU'(x) = |
                   | 0 x<0
Tại x=0, ReLU không khả vi theo nghĩa thông thường; trong thực tế, framework chọn một quy ước cho Gradient tại điểm này.

6.4 Xây dựng một Neural Network hoàn chỉnh
Bây giờ chúng ta kết hợp các neuron thành một mạng Neural Network.
Xét kiến trúc:
- 2 Input Neurons
- 2 Hidden Neurons
- 1 Output Neuron

Đây là mạng có kiến trúc 2 -> 2 -> 1

Bước 1: Hidden Layer
Hai đầu vào:    x1, x2
Hidden Neuron thứ nhất:
        z1 = w1x1 + w2x2 + b1
        
        h1 = ReLU(z1)

Hidden Neuron thứ 2:
        z2 = w3x1 + w4x2 + b2

        h2 = ReLU(z2)

Như vậy, Hidden Layers nhận 2 giá trị đầu vào và tạo ra 2 giá trị mới:
        h1, h2

Bước 2: Output Layer
Output Neuron nhận 2 giá trị từ Hidden Layer:
        z3 = w5h1 + w6h2 +b3

Giả sử sử dụng Linear Activation ở Output Layer:
        y^ = z3

Kết quả y^ là giá trị dự đoán của Neural Network.

Bước 3: Loss Function
Giả sử giá trị thực tế là:
        y = 10
Nhưng Neural Network dự đoán:
        y^ = 7

Chúng ta cần một cách đo lường mức độ sai lệch.
Sử dụng Mean Squared Error (MSE) cho một mẫu:
        L = (y^ - y)^2

Thay số: L = (7-10)^2 = 9

Loss càng nhỏ thì dự đoán càng gần giá trị thực tế.

Lưu ý: Với một mẫu dữ liệu, công thức trên là Squared Error; MSE là trung bình Squared Error trên nhiều mẫu.

6.5 Backpropagation - Truyền Gradient ngược qua Neural Network
Chúng ta đã biết Forward Pass tạo ra Prediction và Loss.
Nhưng làm thế nào mô hình biết cần điều chỉnh w1, w2,... như thế nào?

Chúng ta cần tính đạo hàm riêng:
        dL      dL
        --- ,   ---, ...
        dw1     dw2

Đây là Grandient của Loss theo từng Weight.

6.5.1 Bắt đầu từ Output Layer
Ta có:  L = (y^ - y)^2
Đạo hàm riêng Loss theo Prediction:
        dL
        -- = 2(y^-y)=-6
        dy^

Ý nghĩa: Nếu Prediction tăng một lượng nhỏ, Loss sẽ giảm xấp xỉ 6 lần lượng tăng đó tại điểm hiện tại.

5.5.2 Gradient của Output Weights
Ta có:  y^ = w5h1 + w6h2 + b3

Theo đạo hàm:
        dy^
        -- = h1
        dw5

Áp dụng Chain Rule:
        dL   dL   dy^
        -- = -- . --
        dw5  dy^  dw5

Suy ra: dL
        --- = 2(y^-y)*h1
        dw5

Tương tự:
        dL
        --- = 2(y^-y)*h2
        dw6

Với Bias:
        dL
        --- = 2(y^-y)
        db3

Gradient từ Loss được truyền ngược đến các Weight thông qua Chain Rule.

6.5.3 Gradient truyền về hidden Layer
Hidden Layer thứ nhất:
        h1 = ReLU(z1)
Trong đó:
        z1 = w1x1 + w2x2 + b1

Ta muốn tính:
        dL
        ---
        dw1

Nhìn vào đồ thị phụ thuộc:
        w1 -> z1 -> h1 -> y^ -> L

Áp dụng Chain Rule:
        dL  dL  dy^ dh1 dz1
        ---=---.---.---.---
        dw1 dy^ dh1 dz1 dw1

Ta đã biết:
        dL
        -- = 2(y^-y)
        dy^

        dy^
        ---=w5
        dh1

Đạo hàm ReLU:
        dh1
        --- = ReLU'(z1)
        dz1

Cuối cùng:
        dz1
        --- = x1
        dw1

Kết hợp lại:
        dL
        --- = 2(y^-y).w5.ReLU'(z1).x1
        dw1

Đây chính là Backpropagation.

Một điểm cần hiểu rõ: Gradient tại một node là đạo hàm của Loss theo node đó, không đơn thuần là đạo hàm của node trước đó.
Nó thể hiện mức độ ảnh hưởng của node đến Loss cuối cùng.

5.6 Gradient Descent - Cập nhật Weights và Bias
Sau khi Backpropagation tính được Gradient, chúng ta cần sử dụng chúng để cập nhật các tham số.
Đây là nhiệm vụ của Optimizer.
Thuật toán cơ bản nhất là Gradient Descent.

Công thức:              dL
        w    = w    - a --
         new    old     dw

Trong đó:
        w(old): Weight hiện tại
        w(new): Weight mới
        a: learning rate
        dL/dw: Gradient

Tại sao lại trừ Gradient?
Gradient chỉ hướng mà Loss tăng nhanh nhất tại điểm hiện tại. Vì vậy, ta di chuyển theo hướng ngược lại để giảm Loss.

