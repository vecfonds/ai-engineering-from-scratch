A matrix is a machine that reshapes space. 
Ma trận là một cỗ máy định hình lại không gian. 

1. Matrix Transformation là gì?
Công thức tổng quát:
    v' = Mv
Trong đó:
- v: vector ban đầu
- M: ma trận biến đổi
- v': vector sau biến đổi

2. Tại sao chỉ cần biết 2 Basis Vectors?
Trong không gian 2D, chúng ta có 2 vector cơ sở (standard basis):
    e1 = [1,0]  e2 = [0,1]
- e1: hướng đơn vị trên trục X
- e2: hướng đơn vị trên trục Y

Một vector bất kỳ có thể biểu diễn:
    v = xe1 + ye2

Bây giờ giản sử ma trận M biến đổi 2 vector cơ sở:
    Me1 =   2       Me2 =   1
            1               3

Khi đó ma trận M chính là:
    M = 2   1
        1   3

Tại sao? Vì 2 cột của ma trận chính là vị trí mới của 2 vector cơ sở.
Điều này có nghĩa: 
- Nếu biết vector cơ sở thứ nhất đi đâu
- Biết vector cơ sở thứ 2 đi đâu
- Chúng ta có thể xác định được toàn bộ phép biến đổi tuyến tính trong không gian 2D.

3. Bốn phép biến đổi cơ bản
3.1 Rotation - phép xoay
Rotation là phép biến đổi xoay vector quanh gốc toạ độ 1 góc \(\theta\).

3.2 Scaling - phép phóng to và thu nhỏ
Scaling thay đổi kích thước của vector hoặc hình dạng trong không gian.

Ứng dụng trong AI: Scaling có liên hệ với việc thay đổi tỷ lệ dữ liệu và hình ảnh. Tuy nhiên, cần phân biệt phép biến đổi toạ độ hình học với chuẩn hoá đặc trưng (Feature Scaling) trong Machine Learning.

3.3 Shearing - phép biến đổi xiên
Shearing làm nghiêng hình dạng mà không nhất thiết thay đổi diện tích.
Có 2 dạng cơ bản:
- Shearing theo trục X.
- Shearing theo trục Y.

3.4 Reflection - Phép phản chiếu
Reflection tạo ra hình ảnh đối xứng qua một trục.

4. Composition of Transformations
Trong thực tế, chúng ta thường không chỉ thực hiện 1 phép biến đổi mà kết hợp nhiều phép biến đổi liên tiếp.
Thay vì thực hiện từng bước riêng biệt, ta có thể kết hợp chúng thành một ma trận duy nhất.

Trong phép nhân ma trận, thứ tự thực hiện có ý nghĩa.
- SR: Rotation trước, Scaling sau.
- RS: Scaling trước, Rotation sau.
Đây là tính chất Non-commutative (không có tính giao hoán) của phép nhân ma trận.

5. Eigenvalues và Eigenvectors
5.1 Eigenvectors - Vector riêng
Hãy tưởng tượng một ma trận biến đổi mọi vector trong không gian.
Thông thường, khi một vector đi qua ma trận:
- độ dài có thể thay đổi
- hướng có thể thay đổi
Tuy nhiên, có một số vector đặc biệt mà sau khi biến đổi, chúng vẫn giữ nguyên phương.
=> những vector đó gọi là Eigenvectors

Công thức:      Av = λv
- A: Ma trận biến đổi
- v: Eigenvector
- λ: Eigenvalue
- Av: vector sau biến đổi

Eigenvalue cho biết vector riêng được co giãn bao nhiêu lần (có thể đảo hướng nếu giá trị âm).

- Eigenvalue cho biết mức độ co giãn.
- Eigenvector cho biết hướng xảy ra sự co giãn đó.

5.2 Tìm Eigenvalues như thế nào?
Ta sử dụng phương trình đặc trưng:
        det(A-λI) = 0
Trong đó:
- det: Determinant (định thức)
- I: Identity Matrix (ma trận đơn vị)
- λ: giá trị cần tìm

Với ma trận:
        A = a b
            c d

Phương trình đặc trưng: λ2 - (a+d)λ +(ad-bc) = 0

5.3 Eigendecomposition
Eigendecomposition là kỹ thuật phân rã một ma trận thành 3 thành phần:
    A = VDV-1

Trong đó:
- V: ma trận chứa các Eigenvectors theo từng cột
- D: Ma trận đường chéo chứa các Eigenvalues
- V-1: ma trận nghịch đảo của V

Ý nghĩa hình học:
- chuyển vector sang hệ toạ độ Eigenvectors.
- Scaling theo từng hướng Eigenvectors.
- Chuyển ngược về hệ toạ độ ban đầu.

Không phải mọi ma trận đều phân rã được theo cách này; ma trận cần có đủ số Eigenvectors độp lập tuyến tính.

6. Determinant - định thức
Determinant cho biết diện tích hoặc thể tích thay đổi bao nhiêu lần sau phép biến đổi.

7. Liên hệ với AI Engineering
Khái niệm               Ứng dụng trong AI
Matrix Transformation   Biến đổi dữ liệu, biểu diễn tuyến tính
Rotation/Scaling        Data Augmentation, xử lý hình ảnh
Eigenvectors            PCA, dimensionality reduction
Eigenvalues             Đo phương sai trong PCA
Determinant             phân tích tính khả nghịch, biến đổi không gian
Eigendecomposition      Phân tích ma trận covariance

Ví dụ: PCA
PCA (principal component analysis) là thuật toán giảm chiều dữ liệu.
Giả sử bạn có dữ liệu với 100 features. Bạn muốn giảm xuống còn 2 hoặc 3 đặc trưng nhưng vẫn giữ lại phần lớn thông tin.

PCA sử dụng Eigenvectors và Eigenvalues của ma trận covariance:
- Eigenvectors xác định các hướng dữ liệu quan trọng
- Eigenvalues cho biết lượng phương sai theo từng hướng.
- Chọn các Eigenvectos có Eigenvalues lớn nhất để giảm chiều dữ liệu.
