- Every AI model is just matrix math wearning a fancy hat
Mọi mô hinhf AI chỉ là toán học ma trận đội 1 chiếc mũ lạ mắt

1. Vector
1.1 Khái niệm
Trong lập trình, bạn có thể hình dung Vector là một danh sách (list) các con số.
Trong toán học, Vector không chỉ là danh sách số mà còn biểu diễn một điểm hoặc một hướng trong không gian (tọa độ trong không gian).

1.2 Ứng dụng trong không gian
Trong AI, vector đại diện cho mọi thứ:
- một từ (máy tính không tự hiểu được ý nghĩa của những từ này, chuyển chúng thành các vector số - embedding, các vector được đặt trong một không gian nhiều chiều để biểu diễn đặc trưng hoặc ý nghĩa)
- một hình ảnh (một vector có hàng triệu giá trị pixel)
- một người dùng

2.3 Magnitude - độ lớn của vector
Vector có độ lớn (magnitude), hiểu đơn gian là chiều dài của mũi tên
Magnitude còn được gọi là norm.

2.4 Normalize - Chuẩn hoá Vector
Một vector sau khi normalize sẽ có độ dài bằng 1, nhưng vẫn giữ nguyên hướng
Tại sao cần normalize?
Trong AI, việc chuẩn hoá vector có thể giúp so sánh hướng của các vector mà không bị ảnh hưởng bởi độ lớn của chúng. Đây là cơ sở cho Cosine Similarity (phần Dot Product).

2. Matrix 
MATRICES ARE TRANSFORMATIONS
A matrix transforms one vector into another. It can rotate, stretch, or project.

Hầu hết các phép tính bên trong Neural Network đều liên quan đến Matrix và Tensor.

In AI, matrices ARE the model:
Neural network weights → matrices that transform input into output
Attention scores → matrices that decide what to focus on
Embeddings → matrices that map words to vectors


1. Matrix là gì?
Matrix (ma trận) là một bảng gồm các con số được sắp xếp thành hàng (Rows) và cột (Columns).

2. Matrix trong Machine Learning
Hãy tưởng tượng bạn đang xây dựng 1 hệ thống dự đoán gía nhà.
Ta có thể biểu diễn dữ liệu này bằng Matrix:
- Mỗi Row đại diện cho một căn nhà (một sample)
- Mỗi Column đại diện cho một đăng trưng (feature)
- Mỗi phần tử là giá trị của một feature
Đây là cách dữ liệu được biểu diễn trong Machine Learning.

Matrix(mxn):
- m: số lượng samples
- n: số lượng features

2.3 Các phép toán cơ bản trên Matrix
2.3.1 Matrix Addition - phép cộng ma trận
Hai ma trận có thể cộng trực tiếp khi chúng có cùng Shape.

2.3.2 Scalar Multiplication - nhân ma trận với một số
Scalar là một số đơn lẻ.

2.3.3 Transpose - chuyển vị ma trận
Transpose là phép đổi hàng thành cột và cột thành hàng.
VD: print(A.T)
Lưu ý: Transpose thường được sử dụng để điều chỉnh shape sao cho các phép nhân ma trận hợp lệ.

2.3.4 Matrix Multiplication - phép nhân ma trận
Quy tắc nhân ma trận: cho 2 ma trận A(mxn) và B(pxq)
ta chỉ có thể nhân AxB khi: n=p
Số cột của ma trận bên trái phải bằng số hàng của ma trận bên phải
Nếu hợp lệ, Shape: (mxn)(pxq)=mxq

2.3.5 Hiểu bản chất Row x Column
- lấy một hàng ma trận A
- lấy một cột ma trận B
- nhân từng cặp phần tử tương ứng
- công các kết quả lại

2.3.6 Biểu diễn trong python
VD: C=A@B

2.3.7 Identity Matrix - Ma trận đơn vị
là ma trận vuông có:
- Các phần tử trên đường chéo chính bằng 1
- các phẩn tử còn lại bằng 0

Tính chất quan trọng: A x I = I x A = A
nó tương đương phép nhân một số với 1

Trong python: VD: I = np.eye(3)


3. Dot Product
Vector giúp chúng ta biểu diễn dữ liệu.
Matrix giúp xử lý nhiều dữ liệu cùng lúc.
Dot Product giúp chúng ta tính toán mối quan hệ giữa các vector.

The dot product of two vectors tells you how similar they are.
Tích vô hướng của 2 vector cho biết chúng giống nhau như thế nào.
- Cùng hướng (Same direction): a.b>0 (tương tự - similar)
- Vuông góc (Perpendicular): a.b=0 (không liên quan - unrelated)
- Hướng ngược lại (Opposite direction): a.b<0 (không giống nhau - dissimilar)

This is literally how search engines, recommendation systems, and RAG work -- find vectors with high dot products.
Đây là cách các hệ thống tìm kiếm, hệ thống đề xuất và RAG hoạt động -- tìm vector có tích vô hướng cao.

3.1 Dot Product là gì?
là phép toán giữa hai vector có cùng số chiều, kết quả trả về là một scalar (một số đơn).

Dot product được tính bằng cách:
- nhân các phần tử ở cùng vị trí
- cộng tất cả kết quả lại

3.2 Ý nghĩa hình học
Dot Product có mối liên hệ với góc giữa 2 vector.
Công thức: A.B = \(\|A\|\)\(\|B\|\)\(\cos\theta\)

Trong đó:
- \(\|A\|\): Magnitude của vector A.
- \(\|B\|\): Magnitude của vector B.
- \(\theta\): Góc giữa hai vector.
- \(\cos\theta\): Cosine của góc đó.

Điều này cho chúng ta biết Dot Product phản ánh mức độ cùng hướng của 2 vector, đồng thời còn phụ thuộc vào độ lớn của chúng.

3.3 Ứng dụng Dot Product trong AI Engineering
3.3.1 Embedding và Semantic Search
3.3.2 Attention trong Transformer và LLM
3.3.3 Recommendation System

4. Linear Independence (độc lập tuyến tính)
Vector: biểu diễn dữ liệu bằng các con số
Matrix: tổ chức và xử lý dữ liệu dưới dạng bảng số
Dot Product: tính toán mối quan hệ giữa 2 vector

Linear Independence: nếu có nhiều vector, làm thế nào biết vector nào thực sự mang thông tin mới, và vector nào chỉ là sự kết hợp của những vector khác?

4.1 Linear Independence là gì?
là tính chất của một tập hợp vector mà không có vector nào trong tập hợp đó có thể biểu diễn thành tổ hợp tuyến tính của các vector còn lại.

Note: Hai vector không cần giống hệt nhau mới bị phụ thuộc tuyến tính. Chỉ cần 1 vector có thể biểu diễn từ vector còn lại bằng phép nhân với một scalar.

4.2 Linear Combination - tổ hợp tuyến tính
là việc tạo ra một vector mới bằng cách nhân các vector có sẵn với những hệ số rồi cộng chúng lại.

=> nếu một vector trong tập hợp có thể được tạo thành từ các vector còn lại thì tập hợp đó phụ thuộc tuyến tính.

4.3 Định nghĩa toán học chính xác
Một tập hợp vector: v1, v2, ..., vn
được gọi là Linearly Independent nếu phương trình:

c1v1 + c2v2 + ... + cnvn = 0

chỉ có nghiệm:

c1 = c2 = ... = cn = 0

Nói đơn giản:
- Nếu cách duy nhất để kết hợp các vector tạo ra vector 0 là sử dụng toàn bộ hệ số bằng 0, thì chúng độc lập tuyến tính.
- Ngược lại nếu tồn tại ít nhất một hệ số khác 0 mà vẫn tạo được vector 0, chúng phụ thuộc tuyến tính.

4.4 Vector Space - không gian vector
Trong không gian thực n chiều, số lượng vector độc lập tuyến tính tối đa là n.
VD:
- không gian 1D - số vector độc lập tối đa: 1
- không gian 2D - số vector độc lập tối đa: 2
- không gian 3D - số vector độc lập tối đa: 3
- không gian 100D - số vector độc lập tối đa: 100
=> nếu có 4 vector trong không gian 3 chiều, chắc chắn chúng phụ thuộc tuyến tính.

4.5 Basis - Cơ sở của không gian vector
Basis là tập hợp các vector độc lập tuyến tính và có thể tạo ra toàn bộ không gian vector đang xét.
VD, trong không gian 2D: A = [1,0], B =[0,1]
A và B là một Basis của không gian 2D.
Nhưng basis không nhất thiết phải là hai vector đơn vị này.
VD: A =[1,1] B = [1,-1]
Hai vector này cũng tạo thành một basis của không gian 2D vì chúng độc lập tuyến tính.
=> có thể hình dung Basis như một bộ hướng dẫn cơ bản để biểu diễn mọi vector trong 1 không gian.

Lưu ý: một không gian vector có thể có nhiều Basis khác nhau, nhưng tất cả các Basis của cùng 1 không gian hữu hạn chiều đều có cùng số lượng vector. Số lượng đó chính là Dimension của không gian.

4.6 Matrix Rank - hạng của ma trận
Rank là số lượng vector độc lập tuyến tính tối đa trong các hàng hoặc các cột của một ma trận.

Rank cho biết số chiều độc lập thực sự của dữ liệu được biểu diễn trong ma trận.

=> Đây là điểm rất quan trong khi làm việc với dữ liệu Machine Learning: số lượng features không đồng nghĩa với số lượng thông tin độc lập.

Kiểm tra Rank bằng Numpy: np.linalg.matrix_rank(A)

Numpy sử dụng phương pháp số học dựa trên SVD để ước lượng Rank. Vì vậy, với dữ liệu thực có sai số dấu phẩy động, kết quả có thể phụ thuộc vào ngưỡng số học.

4.7 Determinant - Định ước và linear Independence
Công cụ kiểm tra tính độc lập của ma trận vuông.
Với ma trận vuông, Determinant là 1 công cụ hữu ích để kiểm tra tính độc lập tuyến tính.
Công thức Determinant của ma trận 2x2: det(A)=ad-bc
Công thức Determinant của ma trận 3x3: det(A)=T1-T2, trong đó:
T1: tích của 3 đường chéo xuôi (từ trên trái xuống dưới phải)
T1=aei+bfg+cdh
T2: tích của 3 đường chéo ngược (từ dưới trái lên trên phải)
T2=gec+hfa+idb


Quy tắc:
- Determinant khác 0: các cột (và hàng) độc lập tuyến tính.
- Determinant bằng 0: các cột (và hàng) phụ thuộc tuyến tính.
Quy tắc này áp dụng cho ma trận vuông. Với ma trận không vuông, bạn thường sử dụng Rank.

4.8 Ứng dụng trong AI Engineering
4.8.1 Feature Redundancy - đặc trưng dư thừa
Trong Machine Learning, những features dư thừa có thể gây ra vấn đề như:
- Multicollinearity trong Linear Regression.
- Khó diễn giải hệ số của mô hình.
- Ma trận thiết kế có thể không có full column rank.
- Một số phép tính có thể gặp vấn đề về tính ổn định số.
Lưu ý: Feature tương quan cao không nhất thiết là phụ thuộc tuyến tính hoàn hảo.

4.8.2 PCA - Principal Component Analysis
Principal Component Analysis (PCA) là phương pháp giảm số chiều dữ liệu.
VD: bạn có dataset gồm 100 features. Nhưng nhiều features có thể chứa thông tin trùng lặp hoặc tương quan mạnh.
PCA tìm ra những hướng mới (Principal Components) để biểu diễn dữ liệu với ít chiều hơn, đồng thời giữ lại nhiều phương sai nhất có thể.
Mục tiêu: giảm độ phức tạp nhưng vẫn giữ phần lớn thông tin biến thiên của dữ liệu.
Điều quan trọng: PCA không đơn thuần xoá các vector phụ thuộc tuyến tính. Nó tìm các trục trực giao mới theo thứ tự phương sai giảm dần.

4.8.3 LoRA - Low-Rank Adaptation
LoRA là kỹ thuật giúp Fine-tune mô hình lớn với ít tham số cần huấn luyện hơn.
Thay vì cập nhật trực tiếp toàn bộ Weight Matrix W, LoRA biểu diễn phần thay đổi của trọng số dưới dạng:
\[
\Delta W=BA
\]

\[
\Delta W=BA
\]

Trong đó:
- A và B là các ma trận có Rank thấp.
- Rank \(r\) thường nhỏ hơn nhiều so với kích thước Weight Matrix.
- Mô hình chỉ cần học những ma trận nhỏ hơn thay vì cập nhật toàn bộ trọng số gốc.

5. Basis & Rank (cơ sở và hạng của ma trận)
5.1 Span - không gian sinh
Span của 1 tập vector là tập hợp tất cả các vector có thể tạo ra bằng cách kết hợp tuyến tính các vector trong tập đó.

Span (bao tuyến tính) là TOÀN BỘ KHÔNG GIAN hình học được tạo ra khi bạn kéo giãn và cộng tất cả các vector trong tập hợp gốc lại với nhau theo mọi cách có thể.
 Ví dụ:
 • Nếu các vector gốc cùng nằm trên 1 đường thẳng -> Span là Đường thẳng đó.
• Nếu các vector gốc hợp lực quét được một mặt phẳng -> Span là Mặt phẳng đó.
• Nếu các vector gốc hợp lực quét hết không gian 3D -> Span là Toàn bộ không gian 3D.


5.2 Basis - Cơ sở của không gian vector
Một tập vector được gọi là Basis khi thoã mãn đồng thời hai điều kiện:
- Các vector trong tập độc lập tuyến tính (Linearly Independent)
- Các vector đó sinh ra toàn bộ không gian cần xét (Span)

Basic là một tập sinh tối giản - vừa đủ để span (phủ kín) không gian, không thừa một vector nào.

5.3 Dimension - số chiều của không gian
Dimension là số lượng vector trong 1 basis của không gian vector.
Điểm quan trọng: một không gian có thể có nhiều Basis khác nhau, nhưng số lượng vector trong mỗi Basis luôn bằng nhau.

5.4 Column Space - không gian cột
Column Space là tập hợp tất cả các tổ hợp tuyến tính của các cột trong ma trận A:
    Col(A) = Span(c1,c2,c3)

5.5 Rank - hạng của ma trận
Rank là số lượng vector độc lập tuyến tính tối đa trong các cột (hoặc hàng) của ma trận.

Một số tính chất quan trọng:
- Rank không lớn hơn số hàng
- Rank không lớn hơn số cột
- Rank là số chiều của Column Space
- Số chiều Row Space luôn bằng số chiều của Column Space.

Với ma trận m x n:  0<= Rank(A) <= min(m,n)

Hạng của hàng (Row Rank) = Hạng của cột (Column Rank) 

Rank of a matrix = number of linearly independent columns = number of linearly independent rows. If rank < min(rows, cols), the matrix is rank-deficient. This means:
- The system has infinitely many solutions (or none)
- Information is loss in the transformation
- The matrix cannot be inverted



Hãy hình dung ma trận dữ liệu X có m hàng (mẫu) và n cột (đặc trưng).

Rank là gì? Là số lượng cột (hoặc hàng) độc lập tuyến tính, tức là lượng “thông tin thật sự khác nhau” trong dữ liệu.

a. Full rank (rank = min(m, n))

Ma trận có hạng tối đa, không có đặc trưng thừa.

Nếu số hàng ≥ số cột (nhiều mẫu hơn đặc trưng, trường hợp bình thường): nghiệm bình phương tối thiểu duy nhất. Mô hình học được một bộ trọng số rõ ràng.
Nếu số hàng < số cột (ít mẫu hơn đặc trưng): có vô số nghiệm khớp chính xác dữ liệu, vì có nhiều ẩn hơn phương trình. Đây là tình huống dễ overfit.

b. Rank deficient (rank < min(m, n))

Có đặc trưng dư thừa, ví dụ cột C = 2×A + B, hoặc có hai cột giống hệt nhau (hay gặp khi one-hot encoding mà quên bỏ một cột, gọi là dummy variable trap). Hệ quả: vô số bộ trọng số cho cùng kết quả dự đoán, nên không biết chọn bộ nào. Cần regularization (ridge, lasso...) để chọn ra một nghiệm hợp lý.

c. Rank 1

Trường hợp cực đoan: mọi cột đều là bản sao nhân hệ số của cùng một vector. Toàn bộ dữ liệu nằm trên một đường thẳng, chỉ có một chiều thông tin thật.

d. Near rank-deficient (gần suy biến)

Về lý thuyết ma trận vẫn full rank, nhưng một số singular value rất nhỏ, tức là có các cột gần như phụ thuộc nhau (ví dụ “chiều cao tính bằng cm” và “chiều cao tính bằng inch” cộng thêm chút nhiễu).

Ma trận bị ill-conditioned (condition number rất lớn).
Nhiễu nhỏ ở đầu vào làm nghiệm thay đổi rất lớn, mô hình kém ổn định.
Cách xử lý: SVD truncation (cắt bỏ các singular value nhỏ) hoặc ridge regression (thêm λI để ổn định).

Ví dụ nhanh: dự đoán giá nhà với các đặc trưng diện tích (m²), diện tích (feet²), số phòng.

Hai cột diện tích là bội số của nhau → rank deficient, trọng số không xác định duy nhất.
Nếu hai cột chỉ gần tỉ lệ (do làm tròn) → near rank-deficient, trọng số rất bất ổn.

Ý chính cần nhớ: rank thấp hoặc gần thấp nghĩa là đặc trưng bị trùng lặp, dẫn đến nghiệm không duy nhất hoặc không ổn định, và cách chữa chung là regularization hoặc giảm chiều.

5.6 Ứng dụng trong AI Engineering và Machine Learning
5.6.1 Feature Redundancy - Đặc trưng dư thừa
Ví dụ: dataset có 3 features, feature3 = 2 feature2
Vì vậy nó không cung cấp thông tin tuyến tính độc lập mới.
Điều này có thể khiến mô hình Linear Regression gặp vấn đề khi xác định hệ số duy nhất cho các features.
Rank giúp nhận diện số chiều thông tin tuyến tính độc lập trong ma trận dữ liệu.

5.6.2 PCA - Principal Component Analysis
PCA là phương pháp giảm số chiều dữ liệu.
VD: dataset có 100 features, nhưng nhiều features có thông tin trùng lặp hoặc tương quan mạnh.
PCA tìm các hướng biến thiên chính để biểu diễn dữ liều bằng ít chiều hơn, trong khi cố gắng giữ lại nhiều phương sai nhất có thể.
5.6.3 LoRA - Low-Rank Adaptation
Trong Fine-tuning LLM, LoRA tận dụng ý tưởng ma trận có hạng thấp để giảm số lượng tham số cần huấn luyện.
Loravbieeur diễn cập nhật các trọng số bằng tích 2 ma trận nhỏ, có rank thấp.