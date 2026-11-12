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
