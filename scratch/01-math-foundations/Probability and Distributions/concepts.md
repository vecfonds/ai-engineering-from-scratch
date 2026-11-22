Probability is the language AI uses to express uncertainty.
Xác suất là ngôn ngữ AI sử dụng để thể hiện sự không chắc chắn.

1. Probability Fundamentals - Nền tảng xác suất
1.1 Xác suất giải quyết vấn đề gì?
Hãy tưởng tượng chúng ta xây dựng một mô hình Machine Learning để phân loại email.

Khi nhận được một email, mô hình đưa ra kết quả:
spam        : 0.85
không spam  : 0.15
Mô hình không khẳng định chắc chắn email là spam. Nó thể hiện mức độ tin tưởng thông qua xác suất.

Probability là cách biểu diễn và định lượng sự không chắc chắn (uncertainty).
Trong Machine Learning, chúng ta thường không chỉ quan tâm đến 1 kết quả mà còn quan tâm đến khả năng xảy ra của nhiều kết quả khác nhau.

1.2 Sample Space - không gian mẫu
Sample Space (ký hiệu S) là tập hợp tất cả kết quả có thể xảy ra của một phép thử ngẫu nhiên.
VD: tung một đồng u
Có 2 khả năng:
- H (Head): ngửa
- T (Tail): sấp

Không gian mẫu: S = {H,T}

VD: Tung một viên xúc xắc.
    S = {1,2,3,4,5,6}
Có 6 kết qủa có thể xảy ra.

1.3 Event - Biến cố
Event là 1 tập con của Sample Space, đại diện cho 1 hoặc nhiều kết qủa mà chúng ta quan tâm.

VD: Tung 1 viên xúc xắc.
Không gian mẫu:
    S = {1,2,3,4,5,6}

Chúng ta quan tâm đến biến cố A: xuất hiện số chẵn.
Khi đó:
    A={2,4,6}

Vì vậy:
- S: tất cả kết quả có thể xảy ra
- A: Những kết quả thoả mãn điều kiện chúng ta quan tâm.

1.4 Probability - Xác suất
Xác suất của 1 biến cố A được ký hiệu:
    P(A)
Với một viên xúc xắc công bằng, xác suất xuất hiện chẵn là:
          Số kết quả thuận lợi
    P(A)=----------------------
         Tổng số kết quả có thể
          3  
        =---=0.5=50%
          6

Có nghĩa là nếu tung xúc xắc nhiều lần, tỷ lệ xuất hiện số chẵn sẽ tiến gần 50%.

Ba quy tắc cơ bản của xác suất:
    Quy tắc 1: Xác suất không âm
        P(A)>=0
    Quy tắc 2: Tổng xác suất của toàn bộ không gian mẫu bằng 1
        P(S)=1
        Tức là chắc chắn có một kết quả nào đó xảy ra.
    Quy tắc 3: Hai biến cố không thể xảy ra đồng thời
        Nếu A và B loại từ nhau:
        P(A hoặc B) = P(A)+P(B)
        
        VD: Tung xúc xắc, xác suất ra số 1 hoặc số 2:
        P(1 hoặc 2) = 1/6 + 1/6 = 1/3

1.5 Liên hệ với Machine Learning
Giả sử mô hình phân loại 1 hình ảnh thành 3 lớp:
    P=[0.03, 0.91, 0.06]
Tương ứng:
    Cat: 3%
    Dog: 91%
    Bird: 6%

Chúng ta có:
    0.03+0.91+0.06=1
Đây là một Probability Distribution (phân phối xác suất).

Mô hình dự đoán Dog vì đây là lớp có xác suất cao nhất.
Bạn sẽ gặp lại khái niệm này khi học Softmax và Classification.

