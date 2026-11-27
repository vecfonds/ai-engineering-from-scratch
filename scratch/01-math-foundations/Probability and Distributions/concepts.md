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

5. Common Probability Distributions - Các phân phối xác suất phổ biến
Probability Distribution mô tả cách xác suất được phân bố trên các giá trị mà 1 Random Variable có thể nhận.

5.1 Bernoulli Distribution
A. Bernoulli Distribution mô tả một phép thử ngẫu nhiên chỉ có 2 kết quả có thể xảy ra:
    - Success (thành công): X = 1
    - Failure (thất bại):   X = 0

VD:
    Tung đồng xu: ngửa hoặc sấp
    Email: Spam hoặc Not Spam
    Khách hàng: Mua hàng hoặc không mua hàng
    Model: dự đoán đúng hoặc dự đoán sai

Giả sử chúng ta có một đồng xu không cân bằng:
- Xác suất xuất hiện mặt ngửa: 0.7
- Xác suất xuất hiện mặt sấp : 0.3

Ta có:
    P(X=1)=0.7
    P(X=0)=0.3

Ký hiệu:
    X ~ Bernoulli(p)

Trong đó o là xác suất thành công.

B. Công thức PMF
Công thức tổng quát:
              x      1-x
    P(X=x) = p  (1-p)

Với: x {0,1}

Công thức trên gộp hai trường hợp vào một biểu thức duy nhất.

C. Expected Value và Variance
Expected value:
    E[X] = p
Variance:
    Var(X) = p(1-p)

Lưu ý: Expected Value là giá trị trung bình theo lý thuyết khi thực hiện phép thử rất nhiều lần, không có nghĩa một lần tung đồng xu sẽ cho kết quả 0.7

D. Ứng dụng trong ML
Giả sử xây dựng mô hình dự đoán khách hàng có mua hàng hay không.
Model trả về:
    P(X=1)=0.85
    P(X=0)=0.15

Có nghĩa mô hình dự đoán xác suất khách hàng mua hàng là 85%.
Đây là một bài toán Binary Classification.

5.2 Categorical Distribution
A. Categorical Distribution là gì?
Khi có nhiều hơn 2 kết quả.

VD: xây mô hình phân loại hình ảnh thành:
    Cat
    Dog
    Bird
Chúng ta có 3 kết quả có thể xảy ra.

Giả sử mô hình dự đoán:
Class           Probability
Cat             0.6
Dog             0.3
Bird            0.1
Tổng            1.0

Đây là một Categorical Distribution.

Ký hiệu:
    X ~ Categorical(p1, p2,...,pk)

Trong đó:
    k: số lượng lớp
    p(i): xác suất của lớp thứ i

Điều kiện:
    Tổng xác suất = 1

B. PMF của Categorical Distribution
Giả sử có 3 lớp:
    P=[0.6,0.3,0.1]
Ta có:
    P(X=Cat)=0.6
    P(X=Dog)=0.3
    P(X=Bird)=0.1

Khác với Bernoulli, biến ngẫu nhiên có thể nhận một trong k giá trị phân loại.

C. Ứng dụng trong DL.
Khi Neural Network thực hiện Multi-class Classificaiton, đầu ra thường được chuyển thành xác suất bằng Softmax.
Neural Network không trực tiếp đưa ra kết luận Cat. Nó tạo ra các giá trị đầu ra, sau đó Softmax chuyển chúng thành phân phối xác suất.
Đây cũng là lý do Categorical Distribution rất quan trọng khi học LLM: mô hình ngôn ngữ dự đoán phân phối xác suất trên tập token có thể xuất hiện tiếp theo.

D. Bernoulli vs Categorical
Tiêu chí        Bernoulli       Categorical
Số kết quả      2               k kết quả
VD              Spam/Not spam   Cat/Dog/Bird
Tham số         p               p1,p2,...,pk
Tổng xác suất   1               1
Ứng dụng        Binary          Multi-class Classification 
                Classification

Ghi nhớ: Bernoulli là trường hợp đặc biệt của Categorical Distribution khi chỉ có 2 lớp dữ liệu.

5.3 Uniform Distribution - Phân phối đều
A. Uniform Distribution là gì?
Uniform Distribution mô tả tình huống mà các kết qủa trong một phạm vi có xác suất như nhau (đối ới rời rạc) hoặc có mật độ xác suất như nhau (đối với liên tục).

Có 2 loại:
- Discrete Uniform Distribution
- Continuous Uniform Distribution

B. Discrete Uniform Distribution
VD: Tung 1 viên xúc xắc công bằng.
Các kết quả từ 1 đến 6 đều có xác suất bằng nhau:
    P(X=x)=1/6
Tổng xác suất = 1
Đây là Discrete Uniform Distribution.

C. Continuous Uniform Distribution
Giả sử thời gian xử lý 1 request được mô hình hoá đồng đều trong khoảng 0 đến 10 giây.
Khi đó:
    X ~ Uniform(a,b)
Với:
    a=0, b=10

PDF:
    f(x)=0.1
Mât độ xác suất tại mọi điểm trong khoảng [0,10] đều bằng 0.1
Giả sử muốn tính xác suất request được xử lý từ 2 đến 5 giây:
    P(2<=X<=5) = đạo hàm từ 2 đến 5 của 0.1
               = 0.1(5-2)=0.3
Vậy xác suất là 30%

D. Expected Value và Variance
Đối với Continuous Uniform Distribution:
Expected Value:
          a+b
    E[X]=-----
           2

Variance:
            (b-a)^2
    Var(X)=---------
               12

VD với khoảng [0,10]:
          0+10
    E[X]=------=5
            2
             (10-0)^2
    Var(X) =--------- = 8.33
                12

E. Ứng dụng trong AI
Uniform Distribution thường được sử dụng trong:
    Random initialization
    Random sampling
    Data augmentation
    Monte Carlo Simulation

VD, khi khởi tạo 1 số tham số hoặc sinh dữ liệu ngẫu nhiên trong 1 khoảng, chúng ta có thể lấy mẫu từ Uniform Distribution.

import numpy as np

samples = np.random.uniform(
    low=0,
    high=10,
    size=5
)

print(samples)

Mỗi lần chạy, chương trình sẽ tạo ra 5 giá trị ngẫu nhiên trong khoảng từ 0 đến 10.

5.4 Normal Distribution - Gaussian Distribution
Normal Distribution (phân phối chuẩn) là một phân phối xác suất liên tục có dạng hình chuông (Bell Curve).

Nó xuất hiện trong rất nhiều hiện tượng tự nhiên và các mô hình thống kê.
VD:
    Chiều cao con ngừoi
    sai số đo lường
    nhiễu trong dữ liệu
    sai số dự đoán của một số mô hình

Hãy tưởng tượng chúng ta đo chiều cao của 10000 ngừoi.

Thông thường:
    phần lớn mọi người có chiều cao gần mức trung bình
    Số ngừoi quá thấp ít hơn
    Số ngừoi qúa cao cũng ít hơn.

Dữ liệu có thể tạo thành hình dạng gần giống đường cong hình chuông.

B. Công thức PDF
C. Mean - Trung bình
Mean quyết định vị trí trung tâm của phân phối.
D. Standard Deviation - Độ lệch chuẩn
Standard Deviation quyết định độ phân tán của dữ liệu quanh Mean.
    Sigma nhỏ: dữ liệu tập trung gần Mean
    Sigma lớn: dữ liệu phân tán rộng hơn

Cả 2 có Mean bằng 170 nhưng độ phân tán khác nhau.

E. Empirical Rule - Quy tắc 68-95-99.7
Đây là quy tắc rất quan trọng khi làm việc với Normal Distribution.

F. Standard Normal Distribution
Một trường hợp đặc biệt của Normal Distribution là Standard Normal Distribution:
    Z ~ N(0,1)

G. Ứng dụng trong DL
Normal Distribution xuất hiện trong nhiều khía cạnh:
- Weight Initialization: một số phương pháp khởi tạo trọng số sử dụng phân phối chuẩn.
- Gaussian Noise: Thêm nhiễu vào dữ liệu hoạc đầu vào mô hình.
- Variational Autoencoders (VAE): thường mô hình hoá latent variables bằng Gaussian Distribution.
- Probabilistic Models: mô hình hoá dữ liệu và sự không chắc chắn.

5.5 Poisson Distribution - Phân phối Poisson
A. Poisson Distribution là gì?
Poisson Distribution mô tả xác suất xảy ra 1 số lượng sự kiện nhất định trong một khoảng thời gian hoặc không gian cố định, khi các sự kiện xảy ra độc lập với 1 tốc độ trung bình ổn định.

VD:
    Số request đến server trong 1 giây.
    Số khách hàng đến cửa hàng trong 1 giờ
    Số lỗi xảy ra trong 1 ngày
    Số email nhận được trong 1 phút
Giả sử 1 server trung bình nhận được 3 request mỗi giây.
Chúng ta muốn tính xác suất servẻ nhận đúng 5 request trong 1 giây.
Đây là 1 bài toán phù hợp với Poisson Distribution.

B. Công thức PMF
Ký hiệu:
    X ~ Poisson(lambda)
Trong đó:
    X: số lần sự kiện xảy ra
    lambda: số sự kiện trung bình trong khoảng thời gian hoặc không gian đang xét.

D. Expected Value và Variance
Đây là điểm đặc biệt của Poisson Distribution:
    E[X]= lambda
    Var(X)= lambda

F. Ứng dụng trong AI và Engineering
Poisson Distribution có thể được sử dụng để:
- Mô hình hoá số lượng sự kiện xảy ra
- Phân tích lưu lượng request
- Phát hiện bất thường (Anomaly Detection)
- Phân tích tần suất lỗi
- Mô hình hoá số lượng sự kiện trong dữ liệu

Lưu ý: Poisson thường phù hợp khi các sự kiện xảy ra độc lập và tốc độ trung bình ổn định trong khoảng đang xét. Dữ liệu thực tế không phải lúc nào cũng thoả mãn các giải định này.

5.6 Tổng hợp 5 phân phối xác suất

Distribution    Loại                        Đặc điểm            Ứng dụng
Bernoulli       Discrete                    2 kết quả           Binary Classification
Categorical     Discrete                    k kết quả           Multi-class Classification
Uniform         Discrete/Continuous         Đồng đều            Random Sampling
Nornal          Continuous                  Hình chuông         Weight Initialization, Gaussian Noise
Poisson         Discrete                    Số lần sự kiện      Request Counting, Anomaly Detection

Cách lựa chọn Distribution:
    Chỉ có 2 kết qủa: 0 hoặc 1? Bernoulli
    Có nhiều lớp phân loại khác nhau?   Categorical
    Các giá trị trong 1 khoảng đồng đều?    Uniform
    Dữ liệu tập trung quanh giá trị trung bình, có dạng hình chuông?    Normal
    Số lần sự kiện xảy ra trong 1 khoảng cố định?   Poisson

6. Expected Value & Variance - Giá trị kỳ vọng và phương sai
Hai khái niệm quan trọng để hiểu cách dữ liệu phân bổ, đánh giá sự biến thiên và nền tảng của nhiều thuật toán ML.

Expected Value (E): cho biết giá trị trung bình mà chúng ta kỳ vọng nhận được.
Variance (Var): cho biết dữ liệu phân tán quanh giá trị trung bình nhiều hay ít.

6.1 Expected Value - Giá trị kỳ vọng
A. Expected Value là gì?
Nếu bạn tung xúc xắc 1 lần, bạn không biết kết quả sẽ là bao nhiêu/
Nhưng nếu tung hàng nghìn lần, giá trị trung bình của các kết quả sẽ tiến gần đến:
    1+2+3+4+5+6
    -----------=3.5
        6

Giá trị 3.5 chính là Expected Value.

Ký hiệu:
    E[X]

Điều quan trọng: Expected Value không nhất thiết phải là 1 kết quả có thể xuất hiện trong thực tế.
VD: khi tung xúc xắc, không thể xuất hiện 3.5 nhưng Expected Value vẫn bằng 3.5

B. Công thức tính Expected Value cho Discrete Random Variable
Với biến ngẫu nhiên rời rạc:
    E[X]=tổng xP(X=x)

Có thể hiểu đơn giản:
    Expected Value = Tổng của (giá trị x xác suất xuất hiện giá trị đó)

C. Công thức Expected Value cho Continuous Random Variable
Với biến ngẫu nhiên liên tục, chúng ta thay phép tổng bằng tích phân:

    E[X]=tích phân từ âm vô cực đến dương vô cực của xf(x)
Trong đó:
    x: giá trị của biến ngẫu nhiên
    f(x): PDF của biến ngẫu nhiên

D. Expected Value của các Distribution đã học
Bernoulli
    p

Categorical
    tổng pi xi

Discrete Uniform  
    a+b
    ---
     2

Continuous Uniform
    a+b
    ---
     2

Normal
    Muy

Poisson
    Lambda

6.2 Variance - Phương sai
A. Tại sao cần Variance?
Variance đo lường mức độ phân tán của các giá trị xung quanh Expected Value.

B. Xây dựng công thức Variance
Bước 1: tính độ lệch so với Mean
    X-Mu

Bước 2: bình phương độ lệch
Nếu cộng các độ lệch ban đầu, các giá trị âm và dương triệt tiêu nhau.
Do đó, chúng ta bình phương để loại bỏ dấu âm và làm nổi bật những độ lệch lớn.

Bước 3: tính trung bình bình phương độ lệch

C. Công thức Variance
    Var(X)=E[(X-u)^2]

Viết dứoi dạng tổng:
    Var(X)=tổng (x-u)^2 P(X=x)

trong đó:
    u=E[X]


Với biến ngẫu nhiên liên tục:
    Var(X)=từ âm vô cùng tới dương vô cùng đạo hàm của (x-u)^2 f(x)

D. Công thức rút gọn của Variance

    Var(X)=E[X^2]-(E[X])^2

Chứng minh công thức:
    Var(X)=E[(X-u)^2]
          =E[X^2-2uX+u^2]
          =E[X^2]-2uE[X]+E[u^2]

    Vì u là hằng số và E[X]=u
          =E[X^2]-2u^2+u^2
          =E[X^2]-(E[X])^2

Điểm cần phân biệt:
    E[X^2]: kỳ vọng của bình phương X
    (E[X])^2: bình phương của kỳ vọng X


6.3 Standard Deviation - Độ lệch chuẩn
Variance giúp đo mức độ phân tán, nhưng có 1 nhược điểm: đơn vị của nó bị bình phương.
Điều này không trực quan khi diễn giải dữ liệu.
Vì vậy, chúng ta sử dụng Standard Deviation.

Ký hiệu:
    sigma

Công thức:
    sigma = căn Var(X)

Standard Deviation có cùng đơn vị với dữ liệu ban đầu.

So sánh Mean, Variance và Standard Deviation

Mean                    Trung tâm dữ liệu
Variance                Mức độ phân tán của dữ liệu quanh giá trị trung bình đó, tính bằng bình phương độ lệch
Standard Deviation      Mức độ phân tán của dữ liệu quanh giá trị trung bình đó

Minh hoạ với Normal Distribution
Giả sử:
    X ~ N(100, 10^2)

Ta có:
    Mean=100
    Variance=100
    Standard Deviation=10

Theo quy tắc 68-95-99.7:
    Khoảng 68% dữ liệu nằm trong [90, 110].
    Khoảng 95% dữ liệu nằm trong [80, 120].
    Khoảng 99.7% dữ liệu nằm trong [70, 130].

Lưu ý: Quy tắc này áp dụng cho phân phối chuẩn hoặc dữ liệu có phân phối gần chuản, không phải mọi phân phối.

6.4 Sample Variance và Population Variance
Giả sử bạn muốn tính chiều cao trung bình của toàn bộ sinh viên HCMUT.
Có 2 trường hợp:
    Population (tổng thể): bạn có dữ liệu chiều cao của tất cả sinh viên.
    Sample (mẫu): bạn chỉ thu thập dữ liệu của 500 sinh viên đại diện.
Hai trường hợp này có công thức Variance khác nhau.

A. Population Variance
Khi có toàn bộ dữ liệu của tổng thể:
               1
    sigma^2 = -- Tổng (x-u)^2
               N

Trong đó:
    N: số lượng phần tử trong tổng thể
    u: Mean của tổng thể

B. Sample Variance
Khi chi có 1 mẫu dữ liệu và muốn ước lượng Variance của tổng thể:
             1
    s^2 = ------- tổng (x-x_)^2
            n-1

Trong đó:
    n: số phần tử của mẫu
    x_: Mean của mẫu
    n-1: Degrees of Freedom (bậc tự do)

Tại sao chia cho n-1 thay vì n?
Khi tính Mean từ chính mẫu dữ liệu, chúng ta đã sử dụng 1 thông tin để xác định x_. Điều này khiến các độ lệch quanh Mean mẫu có xu hướng nhỏ hơn độ lệch quanh Mean tổng thể thực sự.
Chia cho n-1 giúp hiệu chỉnh sai lệch ddos, tạo ra một ước lượng không chệch của Population Variance khi các quan sát được lấy mẫu độc lập từ cùng 1 phân phối có phương sai hữu hạn.


7. Joint & Marginal Distributions
Trong Machine Learning, 1 sample thường có nhiều features cùng lúc.
VD, 1 sinh viên:
    X: số giờ học
    Y: số giờ ngủ
    Z: điểm thi
Ta không chỉ muốn biết phân phối của từng biến riêng lẻ, mà còn muốn hiểu các biến này phân phối cùng nhau như thế nào.
Đó chính là mục đích của Joint Distribution.

7.1 Joint Probability Distribution
7.1.1 Joint Distribution là gì?
Joint Probability Distribution (phân phối xác suất đồng thời) mô tả xác suất để nhiều biến ngẫu nhiên đồng thời nhận những giá trị cụ thể.

Với 2 biến ngẫu nhiên X và Y:
    P(X=x, Y=y)
Có thể đọc là:
    Xác suất để X bằng x và Y bằng y

VD:
    X: kết quả tung đồng xu thứ nhất
    Y: kết quả tung đồng xu thứ 2

7.1.2 Biểu diễn bằng Joint Probability Table
Ta có thể biểu diễn phân phối đồng thời bằng bảng:

X\Y     Y=0     y=1
X=0     0.25    0.25
X=1     0.25    0.2

Tổng tất cả xác suất = 1.00

Mỗi ô trong bảng là xác suất để cả 2 biến đồng thời nhận giá trị tương ứng.

VD: P(X=1, Y=0)=0.25
Nghĩa là xác suất đồng xu thứ nhất ngửa và đồng xu thứ 2 sấp là 25%

7.1.3 Điều kiện của Joint Distribution
Tổng tất cả xác suất phải bằng 1.

7.2 Marginal Distribution
7.2.1 Marginal Distribution là gì?
Giả sử chúng ta đã biết phân phối đồngthowif của 2 biến X và Y.
Nhưng bây giờ chỉ myoons biết phân phối của X, không quan tâm đến Y.
Chúng ta cần loại bỏ ảnh hưởng của Y bằng cách cộng xác suất qua tất cả giá trị có thể của Y.
Đó là Marginal Distribution (phân phối biên).

Công thức:
    P (x) = Tổng P(X=x, Y=y)
     X       y

    P (y) = Tổng P(X=x, Y=y)
     Y       x
    
Điểm cần nhớ: Marginal Distribution cho chúng ta biết phân phối của một biến riêng lẻ, bằng cách cộng xác suất trên tất cả khả năng của các biến còn lại.

Với biến liên tục, phép cộng được thay bằng phép tích phân:

    f (x)= tích phân từ âm vô cực đến dương vô cực f   (x,y)dy
     X                                              X,Y

7.3 Conditional Distribution
7.3.1 Conditional Distribution là gì?
Marginal Distribution trả lời:
    X có phân phối như thế nào nếu không xét đến Y?
Conditional Distributioon trả lời:
    X có phân phối như thế nào khi đã biết Y nhận một giá trị cụ thể?

Công thức:
                 P(X=x, Y=y)
    P(X=x|Y=y) = -----------
                    P(Y=y)

Điều kiện là P(Y=y)>0

7.4 Independence - Độc lập giữa các biến
7.4.1 Hai biến độc lập là gì?
Hai biến ngẫu nhiên X và Y độc lập nếu việc biết giá trị của biến này không làm thay đổi phân phối xác suất của biến còn lại.

Công thức:
    P(X=x, Y=y) = P (x) P (y)
                   X     Y

Công thức này phải đúng với mọi cặp giá trị x,y.

7.5 Liên hệ với Machine Learning
7.5.1 Modeling data
Giả sử Dataset có 2 features:
    X: số giờ học
    Y: Điểm thi

Joint Distribution mô tả cách 2 features này xuất hiện cùng nhau:
    P(X,Y)

Marginal Distribution mô tả riêng phân phối số giờ học hoặc điểm thi:
    P(X), P(Y)

7.5.2 Machine Learning Prediction
Giả sử:
    X: thông tin đầu vào của một sample
    Y: nhãn cần được dự đoán
Một mục tiêu quan trọng trong supervised learning là học được:
    P(Y/X)
Nghĩa là xác suất 1 nhãn cụ thể xuất hiện khi đã biết dữ liệu đầu vào.

VD trong bài toán phân loại email:
    P(Spam|Email Content)

Model dự đoán xác suất email thuộc lớp Spam dựa trên nội dung email.

7.5.3 LLM - Next Token Prediction
Trong LLM, mô hình học cách dự đoán token tiếp theo dựa trên các token trước đó:
    P(xt|x1, x2, x3, ... , xt-1)
Đây là một Conditional Distribution.

Tóm tắt:
Joint           Phân phối của nhiều biến đồng thời
Marginal        Phân phối của 1 biến riêng lẻ
Conditional     phân phối của một biến khi biết biến khác
Independence    Hai biến không phụ thuộc xác suất vào nhau


8. Central Limit Theorem (CLT) - Định lý giới hạn trung tâm
Ngay cả khi dữ liệu ban đầu không có phân phối chuẩn (Normal Distribution), trung bình của nhiều mẫu dữ liệu vẫn có xu hướng tuân theo phân phối chuẩn khi kích thước mẫu đủ lớn.

Điều này có ý nghĩa rất lớn trong ML, Data Science, Statistical Inference.

CLT không nói rằng dữ liệu gốc sẽ trở thành phân phối chuẩn. Nó nói rằngphaan phối của các giá trị trung bình sẽ tiến gần đến phân phối chuẩn.

Lưu ý: Sample là 1 tập dữ liệu. Sampling Distribution là phân phối của một thống kê qua nhiều lần lấy mẫu.


9. Log Probabilities, Softmax & Cross Entropy
9.1 Log Probabilities - Logarit của xác suất
Giả sử modeal cần tình xác suất của 1 chuỗi gồm nhiều sự kiện độc lập.
Khi số lượng sự kiện tăng lên, tích các xác suất nhỏ có thể trở nên cực kỳ nhỏ.
Đây là vấn đề Numerical Underflow: máy tính có thể không biểu diễn chính xác những số quá nhỏ.

Giải pháp: sử dụng Logarithm.

9.1.2 Log Probability là gì?
Log Probability đơn giản là lấy logarithm của xác suất:
    logP(A)
Trong ML, thường sử dụng Natual Logarithm (log cơ số e):
    lnP(A)

Điểm cần chú ý:
    Xác suất càng gần 1 thì log càng gần 0
    xác suất càng gần 0 thì log càng âm lớn
    với 0<p<=1, log probability luôn nhỏ hơn hoặc bằng 0

9.1.3 Ứng dụng trong LLM
Giả sử LLM dự đoán 1 câu gồm 4 token:
    I love machine learning

Model dự đoán xác suất từng token dựa trên context trước đó:
    P(I)=0.8
    P(love/I)=0.7
    P(machine/I,love)=0.6
    P(learning/I,love,machine)=0.9

Xác suất của cả chuỗi theo quy tắc chuỗi (chain rule):
    P(sequence)=0.8 x 0.7 x 0.6 x 0.9
               =0.3024
        
Log Probability:
    logP(sequence)=log(0.8)+log(0.7)+log(0.6)+log(0.9)

LLM thường làm việc với log probabilities vì tính toán trên chuỗi dài sẽ ổn định hơn.

9.2 Softmax Function
9.2.1 Vấn đề cần giải quyết
Giả sử một Neural Network thực hiện bài toán phân loại ảnh gồm 3 class:
    Cat
    Dog
    Bird

Sau khi đi qua các layers, output cuối cùng của model là:
    z=[2.0,1.0,0.1]

Những giá trị này được gọi là Logits.

Chúng chưa phải xác suất vì:
    có thể là số âm hoặc số dương bất kỳ
    không nhất thiết nằm trong khoảng [0,1]
    tổng các giá trị không nhất thiết bằng 1

Chúng ta cần chuyển Logits thành Probability Distribution.
Đó là nhiệm vụ của Softmax.

9.2.2 Công thức Softmax
Với vector logits:
    z=[z1,z2,...,zk]

Softmax được định nghĩa:
             e^zi
    P(y=i)= --------
            Tổng e^zj

Trong đó:
    zi: logit của class thứ i
    e^zi: exponential của logit
    Mẫu số: tổng. của exponential của tất cả logits
    K: số class

Softmax chuyển các logits thành xác suất.

9.2.3 Tại sao phải dùng exponential?
Có 3 đặc tính quan trọng:
    e^z luôn dương, do đó xác suất không âm
    giá trị logit lớn hơn sẽ có exponential lớn hơn
    chia cho tổng các exponential đảm bảo tổng xác suất bằng 1

Vì vậy Softmax tạo ra một valid probability distribution.

9.2.4 Tại sao Softmax cần trừ giá trị lớn nhất?
Trong thực tế Logits có thể rất lớn.
Ví dụ:
    z=[1000,1001,1002]
Khi tính exponential trực tiếp, các giá trị có thể vượt quá giới hạn biểu diễn của máy tính.

Để tránh điều này, ta sử dụng:
                 e^(zi-max(z))
    Softmax(zi)= ------------------
                 Tổng e^(zj-max(z))

Đây gọi là Numerically Stable Softmax.

9.3 Cross-Entropy Loss
9.3.1 Model cần học như thế nào?
Giả sử model phân loại ảnh thành 3 class:
    Cat
    Dog
    Bird
Ảnh thực tế là Cat.

Ground Truth (nhãn thực tế) được biểu diễn bằng One-Hot Encoding:
    y=[1,0,0]
Model dự đoán:
    y^=[0.659,0.242,0.099]
Ta muốn model dự đoán xác suất cao cho Cat.
Nhưng làm thế naof để đo model đang dự đoán tốt hay tệ?
Chúng ta cần 1 hàm Loss.

9.3.2 Cross-Entropy là gì?
Cross-Entropy đo mức độ khắc biệt giữa phân phối xác suất thực tế và phân phối xác suất mà model dự đoán.

Công thức:
    L = - Tổng từ i=1 đến K của y log(y^ )
                                 i     i

Trong đó:
    yi: xác suất theo phối phối ground truth
    yi^: xác suất model dự đoán
    K: số class

Với One-Hot Encoding, chỉ có một phần tử yi=1, các phần tử còn lại bằng 0.
Do đó, công thức được rút gọn:
    L = -log(y^   )
              true

Nghĩa là Loss chỉ phụ thuộc vào xác suất model dự đoán cho class thực tế.

9.3.4 Điều gì xảy ra nếu model dự đoán sai?
Model càng tự tin vào class đúng -> Loss càng thấp
Model càng gán xác suất thấp cho class đúng -> Loss càng cao
Nếu xác suất class đúng tiến gần 1 -> Loss tiến cần 0
Nếu xác suất class đúng tiến gần 0 -> Loss tăng rất lớn

9.4.1 Tại sao Cross-Entropy phù hợp với Classification?
Vì Cross-Entropy khuyến khích model tăng xác suất của class đúng.
Trong quá trình Training:
    Neural Network tạo logits
    Softmax chuyển logits thành xác suất
    Cross-Entropy tính Loss
    Backpropagation tính gradient
    Optimizer cập nhật Weights
    Lặp lại quá trình

9.4.2 Một kết quả toán học rất quan trọng
Khi kết hợp Softmax và Cross-Entropy, gradient theo từng logit có dạng rất đơn giản:
    dL
    -- = yi^ - yi
    dzi
Trong đó:
    yi^: xác suất dự đoán
    yi: nhãn thực tế

VD:
    y^=[0.659,0.242,0.099]

    y=[1,0,0]

Gradient:
    dL
    -- = [-0.341, 0.242, 0.099]
    dz

Ý nghĩa:
    Logit của Cat nhận gradient âm -> gradient descent có xu hướng tăng logit Cat
    Logit của Dog và Bird nhận gradient dương -> gradient descent có xu hướng giảm các logits này

Tóm tắt:
Log Probability: biểu diễn xác suất dứoi dạng log, giúp tính toán ổn định
Softmax: chuyển logits thành probability distribution
Cross-Entropy: đo mức độ sai lệch giữa dự đoán và ground truth
Softmax+Cross-Entropy: tổ hợp phổ biến để training classification models

Liên hệ với LLM: LLM sử dụng logits để tạo phân phối xác suất trên vocabulary. Log probabilities được dùng trong tính toán và đánh giá chuỗi token. Cross-Entropy là 1 thành phần cốt lõi trong quá trình huấn luyện mô hình dự đoán token tiếp theo.

