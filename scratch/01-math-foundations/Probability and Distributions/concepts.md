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

2. Conditional Probability - Xác suất có điều kiện

2.1 Conditional Probability là gì?
Tung một viên xúc xắc nhưng cung cấp thêm thông tin:
Kết quả vừa tung là một số lớn hơn 3.
Bây giờ, bạn cần tính xác suất kết quả đó là số chẵn.

Ban đầu, có 6 kết quả có thể xảy ra:
    S={1,2,3,4,5,6}
Nhưng khi biết kết quả lớn hơn 3, chúng ta chỉ còn xem xét:
    B={4,5,6}
Trong phạm vi này, những kết quả chẵn là:
    A={4,6}

Do đó:
    P(A|B)=2/3

Ký hiệu P(A|B) đọc là:
    Probability of A given B
    - Xác suất A xảy ra với điều kiện B đã xảy ra.
    - Xác suất có điều kiện của biến cố A khi biến cố B đã xảy ra.
    - Xác suất A khi biết B đã xảy ra.
    
    Biến cố phụ thuộc: B xảy ra làm thay đổi khả năng xảy ra của A.
    Biến cố độc lập: B xảy ra không ảnh hưởng đến A, khi đó P(A|B)=P(A)

Điểm quan trọng: khi có thêm thông rin, không gian kết quả mà chúng ta xem xét có thể bị thu hẹp lại.

2.2 Công thức Conditional Probability
Công thức tổng quát:
            P(A giao B)
    P(A|B)=-------------
                P(B)
    
Trong đó:
    P(A|B): xác suất A khi biết B xảy ra
    P(A giao B): xác suất cả A và B cùng xảy ra
    P(B): xác suất B xảy ra

Điều kiện:
    P(B)>0

Tại sao phải chia cho P(B)?
Vì chúng ta đang xét xác suất bên trong phạm vi B đã xảy ra, chứ không còn xem toàn bộ không gian mẫu ban đầu.

2.3 Phân biệt P(A|B) và P(B|A)
Hai biểu thức có ý nghĩa khác nhau:
- P(A|B): xác suất A xảy ra khi B đã xảy ra.
- P(B|A): xác suất B xảy ra khi A đã xảy ra.

Chúng không nhất thiết bằng nhau.

Ví dụ trong bài toán dự đoán bệnh:
- A: Một người thực sự mắc bệnh.
- B: Kết quả xét nghiệm dương tính.

Khi đó:
    P(B|A)
Là xác suất xét nghiệm dương tính khi người đó thực sự mắc bệnh.
Biết người đó mắc bệnh → xét nghiệm có dương tính không?

Ngược lại:
P(A|B)
Là xác suất người đó thực sự mắc bệnh khi đã nhận được kết quả xét nghiệm dương tính.
Biết xét nghiệm dương tính → người đó có mắc bệnh không?

2.4 Multiplication Rule – Quy tắc nhân xác suất
Từ công thức Conditional Probability, nhân hai vế với P(B):
    P(A giao B)=P(A|B) x P(B)

Nó cho phép chúng ta tính xác suất hai biến cố cùng xảy ra.

VD: một hộp có 3 viên bi đỏ và 2 viên bi xanh. Lấy ngẫu nhiên 2 viên bi liên tiếp, không hoàn lại.
Gọi:
    A: Lần đầu lấy được bi đỏ
    B: lần thứ 2 lấy được bi đỏ

Ta có:
    P(A)=3/5

Nếu lần đầu đã lấy được bi đỏ, hộp còn 2 bi đỏ trên tổng 4 viên.
    P(B/A)=2/4

Vậy xác suất cả 2 lần đều lấy được bi đỏ:
    P(A giao B)=(3/5) x (2/4) = 0.3

Lưu ý: vì không hoàn lại nên xác suất ở lần thứ 2 phụ thuộc vào kết quả lần thứ nhất.

2.5 Independent Events - Hai biến cố độc lập
Hai biến cố A và B được gọi là độc lập nếu việc xảy ra của biến cố này không ảnh hưởng đến xác suất xảy ra của biến cố kia.

Điều kiện toán học:
    P(A|B)=P(A)

Tương đương:
    P(A giao B) = P(A) x P(B)

VD: Tung đồng xu 2 lần
- A: lần thứ nhất xuất hiện mặt ngửa
- B: lần thứ hai xuất hiện mặt ngửa
Kết qủa lần thứ nhất không ảnh hưởng đến lần thứ 2.
Do đó:
    P(A)=1/2
    P(B)=1/2
    P(A giao B)=(1/2) x (1/2)=1/4

So sánh 2 trường hợp:

                    Có hoàn lại             Không hoàn lại
Lần lấy tiếp theo   Không bị ảnh hưởng      có thể bị ảnh hưởng
Xác suất            thường độc lập          thường phụ thuộc
VD                  Rút bi, trả lại hộp     Rút bi, giữ lại

Lưu ý quan trọng: Hai biến cố có xác suất bằng nhau không có nghĩa là chúng độc lập. Phải kiểm tra điều kiện xác suất ở trên.

3. Bayes' Theorem – Định lý Bayes
Bayes' Theorem giúp chúng ta cập nhật xác suất của một giả thuyết khi nhận được thông tin hoặc bằng chứng mới.

3.1 Tại sao cần Bayes' Theorem?
Ở phần trước, chúng ta đã phân biệt:
    P(A|B)!=P(B|A)

3.2 Công thức Bayes' Theorem
        P(B|A) P(A)
P(A|B)= -----------
            P(B)

Trong đó:
    P(A|B) - Posterior: Xác suất giả thuyết A sau khi có bằng chứng B
    P(B|A) - Likelihood: xác suất quan sát B nếu A đúng
    P(A) - Prior: xác suất ban đầu của A
    P(B) - Evidence: Xác suất quan sát được B

Có thể ghi nhớ:
                Likelihood x Prior
    Posterior = ------------------
                    Evidence
    
Đây là cách diễn giải Bayes' Theorem rất thường gặp trong Machine Learning.

4. PMF vs PDF - Probability Mass Function và Probability Density Function
Hai khái niệm quan trọng giúp chúng ta hiểu cách biểu diễn xác suất cho hai loại dữ liệu:
Discrete Random Variables và Continuous Random Variables.

4.1 Discrete và Continuous là gì?
Discrete Random Variable - biến ngẫu nhiên rời rạc: là biến ngẫu nhiên chỉ nhận những giá trị riêng biệt, có thể đếm được.

VD:
    - Số mặt ngửa khi tung đồng xu 3 lần.
    - Số khách hàng truy cập website.
    - Số email Spam nhận được trong ngày.
    - Số lần 1 token xuất hiện trong văn bản.

Continuous Random Variable - Biến ngẫu nhiên liên tục: Là biến ngẫu nhiên có thể nhận mọi giá trị trong một khoảng liên tục.
VD:
    - Chiều cao của một người
    - Nhiệt độ môi trường
    - Thời gian xử lý một request
    - Trọng lượng của một sản phẩm

Giả sử thời gian xử lý request:
    X thuộc [0,10]
X có thể nhận các giá trị:
    1, 1.5, 1.52, 1.432,...
Không thể liệt kê tất cả các giá trị có thể xảy ra.

4.2 PMF - Probability Mass Function
PMF là hàm xác suất dành cho biến ngẫu nhiên rời rạc.
Nó cho biết xác suất biến ngẫu nhiên X nhận chính xác một giá trị cụ thể.
Ký hiệu:
    PMF = P(X=x)

Tổng xác suất của tất cả giá trị có thể xảy ra phải bằng 1.

Trong ML, PMF rất hữu ích khi mô hình dự đoán các lớp rời rạc như Spam/Not Spam,Card/Dog/Bird,...

4.3 PDF - Probability Density Function
PDF là hàm mật độ xác suất dành cho biến ngẫu nhiên liên tục.
Ký hiệu:
    f(x)
Xác suất của x nằm trong khoảng từ a đến b được tính bằng tích phân.

Lưu ý: PDF có thể lớn hơn 1. Bởi vì PDF biểu diễn mật độ, không phải xác suất tại một điểm.