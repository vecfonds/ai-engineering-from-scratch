Derivatives tell you which way is downhill. That is all a neural network needs to learn.
Đạo hàm cho bạn biết con đường nào đang xuống dốc. Đó là tất cả những gì một mạng lưới thần kinh cần phải học.

Linear Algebra giúp bạn hiểu cách biểu diễn dữ liệu bằng vector và ma trận, thì Calculus giúp bạn hiểu mô hình AI học như thế nào và điều chỉnh các tham số ra sao để dự đoán chính xác hơn.

1. Derivative - đạo hàm
1.1 Tại sao Machine Learning cần đạo hàm?
Hãy tưởng tượng bạn đang xây dựng 1 mô hình dự đoán giá nhà.
Bạn xây dựng mô hình Linear Regression:
    y^ = wx + b
Trong đó:
- x: diện tích nhà
- w: weight (trọng số)
- b: bias (độ lệch)
- y^: giá nhà dự đoán

Giả sử ban đầu:     w = 0.01, b = 0

Với căn nhà 100m2:
                    y^ = 0.01*100 = 1

Mô hình dự đoán 1 tỷ đồng, trong khi giá thực tế là 3 tỷ đồng => Prediction Error
Mô hình đang dự đoán sai. Chúng ta cần điều chỉnh w và b.

Nhưng có một vấn đề:
    Làm sao biết cần tăng hay giảm weight? và nên điều chỉnh bao nhiêu?

=> đây chính là lúc đạo hàm xuất hiện.

1.2 Đạo hàm là gì?
Đạo hàm (Derivative) đo lường tốc độ thay đổi của một hàm số khi đầu vào thay đổi.

VD: f(x) = x2
Một số giá trị của hàm:
    x       f(x) = x2
    0       0
    1       1
    2       4
    3       9
    4       16

Khi x tăng, giá trị f(x) cũng tăng.

Nhưng tốc độ tăng không giống nhau:
- Từ 0 lên 1: kết quả tăng 1.
- Từ 1 lên 2: kết quả tăng 3.
- Từ 2 lên 3: kết quả tăng 5.
- Từ 3 lên 4: kết quả tăng 7.

Có thể thấy tốc độ đang tăng lên.

Đạo hàm giúp chung ta xác định tốc độ thay đổi đó tại một điểm cụ thể.

1.3 Ý nghĩa hình học của đạo hàm
Đạo hàm của một điểm chính là độ dốc (Slope) của đường tiếp tuyến với đồ thị hàm số tại điểm đó.

VD: f(x) = x2
Đạo hàm: f'(x) = 2x

Tại từng điểm:
Vị trí      Đạo hàm     Ý nghĩa
x=-2        -4          Hàm đang giảm
x=-1        -2          Hàm đang giảm
x=0         0           Tiếp tuyến nằm ngang
x=1         2           hàm đang tăng
x=2         4           hàm đang tăng nhanh hơn

Có 3 trường hợp quan trọng:
- f'(x) > 0: hàm số tăng
- f'(x) = 0: tiếp tuyến nằm ngang
- f'(x) < 0: hàm số đang giảm

Lưu ý: Đạo hàm bằng 0 chỉ cho biết tiếp tuyến nằm ngang, chưa đủ để kết luận đó là điểm cực tiểu. 

1.4 Công thức định nghĩa đạo hàm
Về mặt toán học, đạo hàm được định nghĩa bằng giới hạn:
                f(x+h)-f(x)
    f'(x) = lim -----------
            h->0    h

Trong đó:
- f(x): giá trị hàm số tại vị trí ban đầu
- h: một lượng thay đổi rất nhỏ của đầu vào
- f(x+h): giá trị hàm số sau khi đầu vào thay đổi
- phân số: tốc độ thay đổi trung bình
- giới hạn h->0: tốc độ thay đổi tức thời

Đây là cách tính đạo hàm bằng phương pháp số (Numerical Derivative)

1.5 Liên hệ với Machine Learning
Giả sử Loss Function phụ thuộc vào Weight:

        L(w) = (w-3)^2
    
Trong đó:
- w: weight của mô hình
- 3: Weight lý tưởng trong ví dụ đơn giản này
- L(w): sai số của mô hình

Đạo hàm:
            dL
            -- = 2*(w-3)
            dw

Giả sử: w=1
Ta có:
            dL
            -- = 2*(1-3) = -4
            dw

Đạo hàm âm có nghĩa là nếu tăng w một lượng nhỏ thì Loss sẽ giảm (tại vị trí đang xét).
Ngược lại, nếu w = 5:
            dL
            -- = 2*(5-3)=4
            dw

Đạo hàm dương cho biết nếu tăng w thì Loss tăng. Muốn giảm Loss, ta nên giảm w.
Đây là ý tưởng cốt lõi của việc huấn luyện Machine Learning.

Đạo hàm cho biết mức độ và chiều hướng thay đổi của Loss khi thay đổi Weight.

2. Partial Derivatives - Đạo hàm riêng

2.1 Đạo hàm riêng là gì?
Đạo hàm riêng là đạo hàm của 1 hàm nhiều biến theo một biến cụ thể, trong khi giữ các biến còn lại không đổi.
VD: f(x,y) = x^2+3xy+y^2
Hàm số có 2 biến x và y.
Chúng ta có thể tính:
- đạo hàm theo x: giữ y cố định
- Đạo hàm theo y: giữ x cố định

2.2 Cách tính đạo hàm riêng

VD: f(x,y) = x^2+3xy+y^2

Trường hợp 1: Đạo hàm theo x
Quy tắc: Xem y như một hằng số.

Trường hợp 1: Đạo hàm theo y
Quy tắc: Xem x như một hằng số.

Như vậy, cùng 1 hàm số nhưng khi lấy đạo hàm theo những biến khác nhau, chúng ta nhận được những kết quả khác nhau.

2.3 Ứng dụng trong Machine Learning
Hãy quay lại mô hình Linear Regression:

        y^ = w1x1 + w2x2 + b

Giả sử chúng ta dự đoán giá nhà dựa trên:
- x1: diện tích nhà
- x2: số phòng ngủ
- w1, w2: trọng số của từng đặc trưng

Để huấn luyện mô hình, chúng ta cần một Loss Function.
Sử dụng Mean Squared Error (MSE).
Loss Function phụ thuộc vào các Weight.

VD:        L(w1,w2) = (w1-3)^2 + (w2-2)^2
Mục tiêu là tìm w1, w2 để Loss nhỏ nhất.

Bước 1: tính đạo hàm riêng theo w1.
    = 2(w1-3)
Bước 2: Tính đạo hàm riêng theo w2.
    = 2(w2-2)

Gỉa sử: w1=1, w2=4
Ta nhận được:
Với w1    =-4 (nên tăng w1 để giảm Loss)
Với w1    =4  (nên giảm w2 để giảm Loss)

Có thể thấy rằng, mỗi Weight có một đạo hàm riêng tương ứng.
Đó là lý do các mô hình Machine Learning cần Partial Derivatives: chúng ta phải biết từng tham số ảnh hưởng đến Loss như nào.

2.4 Từ Partial Derivatives đến Gradient
Gộp các đạo hàm riêng thành một vector. Vector này gọi là Gradient.
Nói cách khác:
- Partial Derivative: độ thay đổi theo từng biến
- Gradient: tập hợp tất cả đạo hàm riêng thành một vector.

Gradient cho biết hướng tăng nhanh nhất của hàm Loss. Muốn giảm Loss, chúng ta sẽ đi theo hướng ngược lại.

Đây là nền tảng của Gradient Descent.

Tóm tắt:
- Derivative: Đo tốc độ thay đổi của một hàm số.
- Partial Derivatives: Đo tốc độ thay đổi của hàm nhiều biến theo từng biến riêng biệt.

3. Gradient & Gradient Descent
Làm thế nào để mô hình Machine Learning tự điều chỉnh các tham số nhằm giảm sai số dự đoán?
Đó chính là nhiệm vụ của Gradient Descent.

3.1 Gradient

Giả sử chúng ta có hàm Loss:
VD:        L(w1,w2) = (w1-3)^2 + (w2-2)^2

Mục tiêu là tìm w1, w2 sao cho Loss nhỏ nhất.
Từ phần Partial Derivatives, chúng ta tính được đạo hàm riêng của từng biến.
Gộp 2 đạo hàm riêng thành một vector -> đây gọi là Gradient.

Ý nghĩa: Gradient cho biết hướng mà hàm Loss tăng nhanh nhất tại vị trí hiện tại.

VD: tại (w1,w2) = (1,4)
Ta có: nabla L = [-4, 4]
Diễn giải:
- thành phần thứ nhất là -4: tăng w1 sẽ làm loss giảm cục bộ.
- Thành phần thứ hai là +4: giảm w2 sẽ làm Loss giảm cục bộ.

Vì vậy, muốn giảm Loss, chúng ta phải di chuyển ngược với Gradient.

    Descent Direction = -nabla L

3.2 Trực quan hoá Gradient
Hãy tưởng tượng Loss Function là một ngọn đồi hoặc một cái bát hình parabol.
- Độ cao biểu thị giá trị Loss.
- Vị trí trên bề mặt biểu thị các Weight.
- Gradient chỉ hướng dốc lên nhanh nhất.
- Negative Gradient chỉ hướng dốc xuống nhanh nhất.
Mục tiêu của chúng ta là tìm vị trí thấp nhất của bề mặt này.
Đó là điểm mà Loss đạt giá trị nhỏ nhất.

3.3 Gradient Descent là gì?
Gradient Descent (hạ Gradient) là thuật toán tối ưu hoá dùng Gradient để cập nhật các tham số, nhằm tìm giá trị làm Loss giảm dần.

Ý tưởng rất đơn giản:
- Khởi tạo Weight bằng 1 giá trị ban đầu
- Tính Loss
- Tính Gradient
- Di chuyển Weight theo hướng ngược Gradient
- Lặp lại cho đến khi đạt điều kiện dừng

Công thức quan trọng nhất:
    w    = w    - a(nabla L(w))
     new    old

Trong đó:
- w(old): weight hiện tại
- w(new): weight sau khi cập nhật
- a: Learning Rate
- Nabla L(w): Gradient của Loss

Nếu có nhiều Weight, chúng ta cập nhật từng Weight theo Gradient tương ứng.

3.4 Learning Rate là gì?
Learning Rate (tốc độ học) quyết định mỗi lần cập nhật Weight sẽ di chuyển bao xa.
Đây là một Hyperparameter rất quan trọng trong Machine Learning.
Hãy tưởng tượng bạn đang đi xuống một ngọn núi để tìm điểm thấp nhất.
Có 3 trường hợp:
TH1: Learning Rate qúa nhỏ
- mỗi lần cập nhật Weight rất ít
- Loss giảm rất chậm
- Cần nhiều Iterations để tiến gần điểm tối ưu

TH2: Learning Rate phù hợp
- Weight được cập nhật với bước đi hợp lý
- Loss giảm dần
- Có thể hội tụ đến điểm tối ưu

TH3: Learning Rate quá lớn
- Bước cập nhật quá lớn
- Có thể vượt qua điểm tối ưu
- Dao động liên tục hoặc khiến Loss tăng mất kiểm soát

Vì vậy, chọn Learning Rate phù hợp là 1 phần quan trọng khi huấn luyện mô hình.

3.5 Phân biệt Gradient và Gradient Descent

Gradient                                Gradient Descent
là một vector đạo hàm riêng             là thuật toán tối ưu hoá
cho biết hướng tăng nhanh nhất          sử dụng hướng ngược Gradient
là kết quả của phép tính đạo hàm        sử dụng Gradient để cập nhật tham số
không tự cập nhật Weight                cập nhật Weight qua Iterations

Gradient là thông tin về độ dốc. 
Gradient Descent là cách sử dụng thông tin đó để tìm điểm tối ưu.

3.6 Các biến thể của Gradient Descent
Khi làm việc với dữ liệu lớn, chúng ta có nhiều cách tính Gradient.
3.6.1 Batch Gradient Descent
Sử dụng toàn bộ tập dữ liệu để tính Gradient trong mỗi lần cập nhật.
Ưu điểm: Gradient ổn định
Nhược điểm: tốn tài nguyên khi Dataset lớn

3.6.2 Stochastic Gradient Descent (SGD)
Mỗi lần cập nhật chỉ sử dụng một Sample.
Ưu điểm: có thể cập nhật nhanh
Nhược điểm: Gradient dao động nhiều

3.6.3 Mini-batch Gradient Descent
Mỗi lần cập nhật sử dụng một nhóm nhỏ dữ liệu (batch)
- cân bằng giữa tốc độ và độ ổn định
- được sử dụng phổ biến trong Deep Learning

Trong Deep Learning, Mini-batch thường là lựa chọn thực tế vì tận dụng được khả năng xử lý song song của GPU.

