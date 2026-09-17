# BÁO CÁO TỐT NGHIỆP

## XÂY DỰNG HỆ THỐNG PHÁT HIỆN TÉ NGÃ VÀ CẢNH BÁO KHẨN CẤP CHO NGƯỜI CAO TUỔI BẰNG CAMERA, TRÍ TUỆ NHÂN TẠO VÀ IOT

**Trường:** [Tên trường]  
**Khoa/Viện:** [Tên khoa/viện]  
**Ngành:** [Tên ngành]  
**Sinh viên thực hiện:** [Họ và tên sinh viên]  
**Mã số sinh viên:** [MSSV]  
**Giảng viên hướng dẫn:** [Họ và tên giảng viên hướng dẫn]  
**Năm thực hiện:** 2026

---

## LỜI CẢM ƠN

Em xin gửi lời cảm ơn chân thành đến quý thầy cô [Tên khoa/viện] đã tận tình truyền đạt kiến thức, định hướng phương pháp học tập và tạo điều kiện để em có thể hoàn thành đề tài tốt nghiệp này.

Đặc biệt, em xin bày tỏ lòng biết ơn đến thầy/cô [Tên giảng viên hướng dẫn], người đã hướng dẫn, góp ý và hỗ trợ em trong suốt quá trình nghiên cứu, xây dựng hệ thống, thử nghiệm và hoàn thiện báo cáo.

Em cũng xin cảm ơn gia đình, bạn bè và những người đã hỗ trợ em trong quá trình kiểm thử hệ thống, thực hiện các kịch bản demo, góp ý giao diện phần mềm và kiểm tra khả năng cảnh báo của thiết bị phần cứng.

Đề tài được thực hiện trong phạm vi thời gian có hạn, đồng thời bài toán phát hiện té ngã trong môi trường thực tế còn nhiều yếu tố phức tạp như góc camera, ánh sáng, che khuất cơ thể và sự đa dạng hành vi của người dùng. Vì vậy, báo cáo khó tránh khỏi thiếu sót. Em kính mong nhận được sự góp ý của quý thầy cô để đề tài được hoàn thiện hơn.

---

## LỜI CAM ĐOAN

Em xin cam đoan đề tài “Xây dựng hệ thống phát hiện té ngã và cảnh báo khẩn cấp cho người cao tuổi bằng camera, trí tuệ nhân tạo và IoT” là kết quả nghiên cứu, xây dựng và thử nghiệm của bản thân dưới sự hướng dẫn của giảng viên hướng dẫn.

Hệ thống trong đề tài sử dụng một số thư viện và mô hình nền mã nguồn mở như PyTorch, OpenCV, Ultralytics YOLOv8 Pose, face_recognition, Telegram Bot API và các thư viện phục vụ lập trình ESP32. Những thành phần này được sử dụng như công cụ nền tảng. Phần xử lý chuỗi đặc trưng, mô hình phân loại GRU, logic xác nhận té ngã, dashboard web, cơ chế cảnh báo và kết nối phần cứng được xây dựng, tích hợp và hiệu chỉnh trong phạm vi đồ án.

Các số liệu huấn luyện, cấu hình hệ thống và kết quả thực nghiệm được tổng hợp từ project hiện có tại thư mục `c:\DATN`. Em xin chịu trách nhiệm về nội dung trình bày trong báo cáo.

---

## TÓM TẮT

Té ngã là một trong những tai nạn phổ biến và nguy hiểm đối với người cao tuổi. Khi một người lớn tuổi bị ngã, đặc biệt là trong trường hợp sống một mình hoặc không có người chăm sóc bên cạnh, việc phát hiện muộn có thể dẫn đến hậu quả nghiêm trọng như chấn thương, bất tỉnh, mất khả năng tự gọi trợ giúp hoặc nằm bất động trong thời gian dài. Vì vậy, việc xây dựng một hệ thống có khả năng giám sát tự động, phát hiện té ngã và gửi cảnh báo kịp thời có ý nghĩa thiết thực trong chăm sóc sức khỏe và hỗ trợ an toàn tại nhà.

Đề tài này xây dựng một hệ thống phát hiện té ngã từ camera/video, kết hợp giữa trí tuệ nhân tạo, web dashboard, cảnh báo Telegram, xác thực khuôn mặt và thiết bị IoT ESP32. Hệ thống sử dụng YOLOv8 Pose pretrained để phát hiện người và trích xuất các điểm keypoint trên cơ thể. Từ kết quả này, hệ thống xây dựng vector đặc trưng gồm 37 chiều mô tả hình học khung bao, tư thế cơ thể và chuyển động theo thời gian. Các chuỗi đặc trưng được đưa vào mô hình GRU tự huấn luyện để phân loại trạng thái người trong khung hình thành các lớp như đi/đứng, ngồi, cúi, nằm chủ động, đang ngã và đã ngã.

Điểm quan trọng của đề tài là hệ thống không chỉ dựa vào kết quả dự đoán của mô hình trên từng frame. Sau khi mô hình GRU đưa ra xác suất cho từng trạng thái, một state machine tiếp tục kiểm tra các bằng chứng theo thời gian như tốc độ rơi, mức giảm chiều cao cơ thể, tư thế nằm ngang sau ngã, trạng thái bất động và quá trình phục hồi. Cách kết hợp này giúp hệ thống giảm báo nhầm trong các tình huống sinh hoạt thường ngày như ngồi xuống, cúi nhặt đồ, nằm nghỉ chủ động, đi sát camera hoặc mất track trong thời gian ngắn.

Ngoài phần nhận diện, đề tài phát triển một hệ thống cảnh báo hoàn chỉnh. Web dashboard cho phép theo dõi nhiều camera, xem trạng thái từng người, lịch sử cảnh báo, ảnh snapshot và video output. Hệ thống có đăng nhập và xác thực khuôn mặt trước khi truy cập dashboard. Khi phát hiện nguy hiểm, hệ thống gửi cảnh báo Telegram kèm ảnh chụp hiện trường và link xác nhận an toàn. Đồng thời, ESP32 đọc trạng thái từ web server qua endpoint `/status` và phát cảnh báo âm thanh tại chỗ thông qua module I2S/MAX98357A.

Kết quả huấn luyện tốt nhất đạt tại epoch 19 với accuracy validation 60%, F1-score lớp FALLING đạt 0.825 và fall_safety_score đạt 0.761. Kết quả cho thấy hệ thống có khả năng nhận diện tốt giai đoạn đang ngã, vốn là giai đoạn quan trọng để phát hiện sớm sự cố. Tuy nhiên, tập test độc lập hiện còn nhỏ và một số lớp như BENDING, UNCONSCIOUS chưa có đủ mẫu dữ liệu, do đó hệ thống cần tiếp tục được mở rộng dữ liệu và kiểm thử thực tế trước khi triển khai chính thức.

**Từ khóa:** phát hiện té ngã, người cao tuổi, YOLOv8 Pose, GRU, state machine, web dashboard, Telegram, ESP32, IoT, xác thực khuôn mặt.

---

## MỤC LỤC

- Danh mục các từ viết tắt  
1. Chương 1. Tổng quan đề tài  
2. Chương 2. Cơ sở lý thuyết và công nghệ sử dụng  
3. Chương 3. Phân tích và thiết kế hệ thống  
4. Chương 4. Xây dựng và cài đặt hệ thống  
5. Chương 5. Thực nghiệm và đánh giá  
6. Chương 6. Kết luận và hướng phát triển  
7. Tài liệu tham khảo  
8. Phụ lục

---

## DANH MỤC CÁC TỪ VIẾT TẮT

**Bảng 0.1. Danh mục các từ viết tắt sử dụng trong báo cáo**

| Từ viết tắt | Thuật ngữ đầy đủ | Ý nghĩa trong báo cáo |
|---|---|---|
| ADL | Activities of Daily Living | Hoạt động sinh hoạt hằng ngày |
| AI | Artificial Intelligence | Trí tuệ nhân tạo |
| API | Application Programming Interface | Giao diện lập trình ứng dụng |
| CNN | Convolutional Neural Network | Mạng nơ-ron tích chập |
| COCO | Common Objects in Context | Bộ dữ liệu/chuẩn keypoint thường dùng trong thị giác máy tính |
| CV | Computer Vision | Thị giác máy tính |
| ESP32 | Espressif ESP32 | Vi điều khiển có WiFi/Bluetooth, dùng để nhận trạng thái và phát cảnh báo âm thanh |
| FAINT | Faint | Trạng thái ngất hoặc cảnh báo bất tỉnh trong hệ thống |
| FALLEN | Fallen | Trạng thái đã ngã |
| FALLING | Falling | Trạng thái đang ngã |
| FPS | Frames Per Second | Số khung hình trên giây |
| GPIO | General Purpose Input/Output | Chân vào/ra đa dụng của vi điều khiển |
| GRU | Gated Recurrent Unit | Mạng hồi quy có cổng, dùng để xử lý chuỗi đặc trưng theo thời gian |
| GUI | Graphical User Interface | Giao diện đồ họa người dùng |
| HOG | Histogram of Oriented Gradients | Đặc trưng hướng gradient trong xử lý ảnh |
| HTTP | HyperText Transfer Protocol | Giao thức truyền tải dữ liệu giữa trình duyệt, server và thiết bị |
| I2S | Inter-IC Sound | Chuẩn giao tiếp âm thanh số, dùng với module khuếch đại MAX98357A |
| IoT | Internet of Things | Internet vạn vật |
| IP | Internet Protocol | Giao thức địa chỉ mạng, thường dùng trong camera IP |
| JSON | JavaScript Object Notation | Định dạng trao đổi dữ liệu giữa web server, dashboard và ESP32 |
| LSTM | Long Short-Term Memory | Mạng hồi quy nhớ dài ngắn hạn |
| MSSV | Mã số sinh viên | Mã định danh sinh viên |
| OpenCV | Open Source Computer Vision Library | Thư viện xử lý ảnh và thị giác máy tính |
| RNN | Recurrent Neural Network | Mạng nơ-ron hồi quy |
| TTS | Text-to-Speech | Chuyển văn bản thành giọng nói |
| UI | User Interface | Giao diện người dùng |
| URL | Uniform Resource Locator | Địa chỉ tài nguyên trên mạng |
| WiFi | Wireless Fidelity | Kết nối mạng không dây |
| YOLO | You Only Look Once | Mô hình phát hiện đối tượng thời gian thực |
| YOLOv8 | You Only Look Once version 8 | Phiên bản YOLO được dùng để phát hiện người và keypoint tư thế |

---

# CHƯƠNG 1. TỔNG QUAN ĐỀ TÀI

## 1.1. Đặt Vấn Đề

Sự phát triển của công nghệ thông tin, trí tuệ nhân tạo và các thiết bị IoT đã mở ra nhiều hướng tiếp cận mới trong lĩnh vực chăm sóc sức khỏe. Trong đó, giám sát an toàn cho người cao tuổi là một bài toán có ý nghĩa thực tiễn cao. Người cao tuổi thường đối mặt với nguy cơ té ngã do suy giảm sức khỏe, giảm khả năng giữ thăng bằng, bệnh lý nền hoặc môi trường sinh hoạt chưa an toàn. Một cú ngã tưởng chừng đơn giản có thể dẫn đến gãy xương, chấn thương đầu, mất ý thức hoặc các biến chứng nghiêm trọng nếu không được hỗ trợ kịp thời.

Trong các gia đình hiện đại, không phải lúc nào người thân cũng có thể ở bên cạnh để quan sát liên tục. Tại các cơ sở chăm sóc, nhân viên y tế cũng khó theo dõi đồng thời nhiều khu vực trong thời gian dài. Vì vậy, một hệ thống giám sát tự động có khả năng phát hiện té ngã từ camera và gửi cảnh báo nhanh đến người giám sát là rất cần thiết.

Các phương pháp phát hiện té ngã có thể chia thành nhiều nhóm, ví dụ thiết bị đeo, cảm biến môi trường và camera. Thiết bị đeo có thể đo gia tốc hoặc chuyển động cơ thể, nhưng phụ thuộc vào việc người dùng phải đeo đúng cách và thường xuyên. Cảm biến môi trường ít xâm phạm quyền riêng tư hơn nhưng khó mô tả đầy đủ hành vi cơ thể. Camera có ưu điểm là không cần tiếp xúc trực tiếp, dễ triển khai ở nhiều không gian và có thể cung cấp bằng chứng trực quan thông qua ảnh hoặc video.

Tuy nhiên, phát hiện té ngã bằng camera không phải là bài toán đơn giản. Một người nằm nghỉ trên giường hoặc sofa có thể có tư thế giống người đã ngã. Một người cúi nhặt đồ hoặc ngồi xuống nhanh cũng có thể tạo ra chuyển động gần giống giai đoạn ngã. Ngoài ra, góc đặt camera, ánh sáng, người đi sát camera, che khuất cơ thể hoặc mất track tạm thời đều có thể gây báo nhầm. Do đó, hệ thống cần kết hợp cả nhận diện tư thế, phân tích chuyển động theo thời gian và cơ chế xác nhận sự kiện.

Từ những vấn đề trên, đề tài lựa chọn xây dựng một hệ thống phát hiện té ngã dựa trên YOLOv8 Pose, mô hình GRU và state machine, đồng thời tích hợp dashboard web, Telegram và ESP32 để tạo thành một quy trình cảnh báo hoàn chỉnh.

## 1.2. Lý Do Chọn Đề Tài

Đề tài được lựa chọn vì ba lý do chính.

Thứ nhất, đây là một bài toán có tính ứng dụng cao. Hệ thống có thể hỗ trợ giám sát người cao tuổi tại nhà, phòng bệnh, viện dưỡng lão hoặc các khu vực cần theo dõi an toàn. Khi sự cố xảy ra, cảnh báo sớm có thể giúp người thân hoặc nhân viên chăm sóc phản ứng nhanh hơn.

Thứ hai, đề tài phù hợp để vận dụng nhiều kiến thức đã học: xử lý ảnh, học sâu, mô hình chuỗi thời gian, lập trình Python, phát triển web, giao tiếp API, lưu trữ dữ liệu, xác thực người dùng và lập trình vi điều khiển ESP32. Đây không chỉ là bài toán huấn luyện một mô hình AI, mà là bài toán thiết kế và tích hợp một hệ thống hoàn chỉnh.

Thứ ba, hướng tiếp cận của đề tài có điểm thực tế rõ ràng. Hệ thống sử dụng YOLOv8 Pose pretrained để trích xuất tư thế, nhưng không xem YOLO là mô hình phát hiện té ngã. Phần phát hiện té ngã được xây dựng riêng thông qua bộ đặc trưng 37 chiều, mô hình GRU tự huấn luyện và state machine giảm báo nhầm. Ngoài ra, hệ thống có web dashboard, cảnh báo Telegram, xác nhận an toàn và ESP32 phát âm thanh, phù hợp với một luồng demo thực tế từ camera đến người nhận cảnh báo.

## 1.3. Mục Tiêu Đề Tài

Mục tiêu tổng quát của đề tài là xây dựng một hệ thống có khả năng phát hiện té ngã từ camera/video và gửi cảnh báo khẩn cấp đến người giám sát trong thời gian ngắn.

Các mục tiêu cụ thể gồm:

- Xây dựng pipeline phát hiện người và keypoint tư thế bằng YOLOv8 Pose.
- Trích xuất bộ đặc trưng mô tả hình học cơ thể, tư thế và chuyển động theo thời gian.
- Xây dựng và huấn luyện mô hình GRU để phân loại trạng thái người trong video.
- Thiết kế state machine để xác nhận sự kiện té ngã và giảm báo nhầm.
- Xây dựng dashboard web hỗ trợ theo dõi nhiều camera, hiển thị trạng thái, ảnh cảnh báo và video output.
- Xây dựng chức năng đăng nhập và xác thực khuôn mặt cho người giám sát.
- Tích hợp cảnh báo Telegram kèm ảnh snapshot và link xác nhận an toàn.
- Tích hợp ESP32 để nhận trạng thái từ web server và phát cảnh báo âm thanh tại chỗ.
- Đánh giá hệ thống bằng kết quả huấn luyện, validation và các kịch bản demo thực tế.

## 1.4. Đối Tượng Và Phạm Vi Nghiên Cứu

Đối tượng nghiên cứu của đề tài là hành vi té ngã của con người trong video, đặc biệt là người cao tuổi trong bối cảnh cần giám sát an toàn. Hệ thống tập trung vào việc nhận diện các trạng thái cơ bản như đi/đứng, ngồi, cúi, nằm chủ động, đang ngã, đã ngã và bất tỉnh.

Phạm vi thực hiện của đề tài:

- Dữ liệu đầu vào là webcam, camera IP hoặc video local có thể đọc bằng OpenCV.
- Hệ thống sử dụng YOLOv8 Pose pretrained để phát hiện người và trích xuất keypoint.
- Mô hình GRU được huấn luyện trên chuỗi đặc trưng theo thời gian.
- Dashboard web hỗ trợ tối đa 4 camera theo cấu hình hiện tại.
- Cảnh báo được gửi qua dashboard, Telegram và ESP32.
- Xác thực khuôn mặt dùng cho người giám sát trước khi truy cập dashboard.

Những nội dung chưa nằm trong phạm vi chính:

- Chưa triển khai hệ thống trên cloud production hoàn chỉnh.
- Chưa có cơ chế chống giả mạo khuôn mặt nâng cao như liveness detection.
- Chưa đánh giá trên tập test lớn độc lập với nhiều môi trường triển khai thực tế.
- Chưa tối ưu sâu cho các thiết bị edge có tài nguyên rất thấp.

## 1.5. Ý Nghĩa Của Đề Tài

Về mặt kỹ thuật, đề tài thể hiện quá trình xây dựng một hệ thống AI ứng dụng đầy đủ: từ chuẩn bị dữ liệu, trích xuất đặc trưng, huấn luyện mô hình, suy luận thời gian thực, xử lý logic sau mô hình, thiết kế giao diện giám sát đến tích hợp kênh cảnh báo và thiết bị phần cứng.

Về mặt thực tiễn, hệ thống giúp phát hiện sớm các tình huống nguy hiểm, gửi thông tin cảnh báo trực quan đến người thân hoặc nhân viên giám sát, đồng thời phát cảnh báo tại chỗ qua ESP32. Điều này có thể góp phần rút ngắn thời gian phản ứng khi người cao tuổi gặp sự cố.

Về mặt học tập, đề tài giúp sinh viên hiểu rõ hơn rằng một hệ thống trí tuệ nhân tạo trong thực tế không chỉ cần mô hình có kết quả tốt, mà còn cần cơ chế giảm báo nhầm, giao diện dễ dùng, bảo mật truy cập, cảnh báo đáng tin cậy và khả năng vận hành ổn định.

## 1.6. Bố Cục Báo Cáo

Báo cáo gồm sáu chương chính. Chương 1 trình bày tổng quan, lý do chọn đề tài, mục tiêu, phạm vi và ý nghĩa. Chương 2 trình bày cơ sở lý thuyết và công nghệ sử dụng. Chương 3 phân tích yêu cầu và thiết kế hệ thống. Chương 4 mô tả quá trình xây dựng, cài đặt và tích hợp. Chương 5 trình bày kết quả huấn luyện, thực nghiệm và đánh giá. Chương 6 tổng kết kết quả đạt được, hạn chế và hướng phát triển.

---

# CHƯƠNG 2. CƠ SỞ LÝ THUYẾT VÀ CÔNG NGHỆ SỬ DỤNG

## 2.1. Bài Toán Phát Hiện Té Ngã

### 2.1.1. Khái Niệm Té Ngã Trong Xử Lý Video

Phát hiện té ngã là bài toán nhận diện một chuỗi hành vi bất thường của con người. Một sự kiện té ngã thường không chỉ được xác định bởi một tư thế, mà bởi quá trình chuyển đổi từ trạng thái đứng hoặc di chuyển sang trạng thái mất thăng bằng, rơi xuống, nằm ngang và có thể bất động sau đó.

### 2.1.2. Khó Khăn Khi Chỉ Xét Từng Frame

Nếu hệ thống chỉ xét một frame đơn lẻ, rất dễ xảy ra nhầm lẫn. Người đang nằm nghỉ có thể giống người đã ngã. Người cúi sâu hoặc ngồi xuống nhanh có thể giống một phần chuyển động té ngã. Do đó, bài toán cần xem xét cả đặc trưng không gian và đặc trưng thời gian.

### 2.1.3. Hướng Tiếp Cận Của Đề Tài

Trong đề tài này, hệ thống giải quyết bài toán theo ba lớp xử lý:

1. Dùng YOLOv8 Pose để phát hiện người và keypoint.
2. Dùng mô hình GRU để học chuỗi đặc trưng theo thời gian.
3. Dùng state machine để kiểm tra bằng chứng té ngã và giảm báo nhầm.

Cách tiếp cận này giúp hệ thống tận dụng được sức mạnh của học sâu, đồng thời vẫn có các điều kiện logic để kiểm soát những tình huống nhạy cảm trong thực tế.

## 2.2. YOLOv8 Pose

### 2.2.1. Vai Trò Của YOLOv8 Pose

YOLO là họ mô hình phát hiện đối tượng một giai đoạn, nổi bật nhờ tốc độ xử lý nhanh và phù hợp với bài toán thời gian thực. YOLOv8 Pose mở rộng khả năng phát hiện đối tượng bằng cách ước lượng thêm các keypoint trên cơ thể người.

### 2.2.2. Nhiệm Vụ Trong Hệ Thống

Trong đề tài, file model `yolov8n-pose.pt` được sử dụng như model nền pretrained. Vai trò của YOLOv8 Pose là:

- Phát hiện người trong từng frame.
- Trả về bounding box của người.
- Trả về độ tin cậy detection.
- Trả về 17 keypoint cơ thể theo chuẩn COCO.

### 2.2.3. Giới Hạn Của YOLOv8 Pose

Điểm cần nhấn mạnh là YOLOv8 Pose không trực tiếp phát hiện té ngã. Đây chỉ là bước trích xuất thông tin thị giác ban đầu. Phần nhận diện trạng thái té ngã được thực hiện bởi mô hình GRU và state machine do đề tài xây dựng.

## 2.3. Keypoint Tư Thế Người

### 2.3.1. Bộ Keypoint Theo Chuẩn COCO

Keypoint là các điểm đại diện cho vị trí khớp hoặc bộ phận chính trên cơ thể. YOLOv8 Pose sử dụng 17 keypoint theo chuẩn COCO:

**Bảng 2.1. Danh sách các keypoint cơ thể theo chuẩn COCO**

| Nhóm | Keypoint |
|---|---|
| Đầu | mũi, mắt trái, mắt phải, tai trái, tai phải |
| Tay | vai trái, vai phải, khuỷu tay trái, khuỷu tay phải, cổ tay trái, cổ tay phải |
| Thân và chân | hông trái, hông phải, gối trái, gối phải, mắt cá trái, mắt cá phải |

### 2.3.2. Ý Nghĩa Của Keypoint Trong Đề Tài

Từ các keypoint này, hệ thống có thể suy ra nhiều đặc trưng có ích:

- Thân người có đang nằm ngang hay không.
- Vai và hông có thay đổi vị trí theo chiều dọc hay không.
- Chân có bị gập nhiều hay không.
- Tỉ lệ keypoint còn nhìn thấy trong frame.
- Độ tin cậy trung bình của keypoint.

Các đặc trưng này đặc biệt hữu ích để phân biệt giữa người đứng, người ngồi, người cúi và người nằm.

## 2.4. Đặc Trưng Thị Giác Và Chuyển Động

### 2.4.1. Lý Do Sử Dụng Vector Đặc Trưng

Sau khi phát hiện người, hệ thống không đưa trực tiếp ảnh thô vào mô hình GRU. Thay vào đó, hệ thống trích xuất vector đặc trưng 37 chiều. Cách này giúp giảm kích thước dữ liệu, tăng tốc xử lý và tập trung vào các thông tin liên quan trực tiếp đến té ngã.

### 2.4.2. Các Nhóm Đặc Trưng

Các nhóm đặc trưng gồm:

**Bảng 2.2. Các nhóm đặc trưng thị giác và chuyển động được sử dụng**

| Nhóm | Ví dụ đặc trưng |
|---|---|
| Bounding box | tâm box, chiều rộng, chiều cao, diện tích, tỉ lệ width/height |
| Tư thế | độ nằm ngang của thân, độ nằm ngang toàn thân, vị trí vai/hông/gối/mắt cá |
| Keypoint | tỉ lệ keypoint nhìn thấy, độ tin cậy trung bình, chiều rộng/cao theo keypoint |
| Chuyển động | delta tâm, delta chiều cao, tốc độ, tốc độ rơi xuống, tăng diện tích |
| Trạng thái thiếu quan sát | missing_flag khi track tạm thời không có detection |

### 2.4.3. Chuẩn Hóa Tọa Độ

Tất cả các tọa độ được chuẩn hóa theo kích thước frame để giảm phụ thuộc vào độ phân giải camera.

## 2.5. Mô Hình GRU

### 2.5.1. Vai Trò Của GRU

GRU, viết tắt của Gated Recurrent Unit, là một kiến trúc mạng hồi quy dùng để xử lý dữ liệu chuỗi. GRU có khả năng ghi nhớ thông tin từ các bước thời gian trước, phù hợp với bài toán mà trạng thái hiện tại phụ thuộc vào chuyển động trước đó.

### 2.5.2. Dữ Liệu Đầu Vào Của Mô Hình

Trong đề tài, mỗi mẫu đầu vào của mô hình là một chuỗi 48 vector đặc trưng, được lấy mẫu ở 10 FPS. Như vậy, mỗi chuỗi biểu diễn khoảng 4,8 giây chuyển động. Mô hình GRU học sự thay đổi của cơ thể qua thời gian để phân loại trạng thái cuối chuỗi.

### 2.5.3. Cấu Hình Mô Hình

Cấu hình mô hình:

**Bảng 2.3. Cấu hình mô hình GRU dùng trong đề tài**

| Tham số | Giá trị |
|---|---:|
| input_dim | 37 |
| sequence_length | 48 |
| hidden_dim | 128 |
| num_layers | 2 |
| bidirectional | true |
| dropout | 0.30 |
| num_classes | 8 |

### 2.5.4. Ý Nghĩa Của GRU Đối Với Bài Toán Té Ngã

Mô hình được cài đặt trong `phan_mem/src/models.py`, lớp `TemporalGRUFallClassifier`.

GRU là phần quan trọng vì té ngã không phải là một trạng thái đứng yên mà là một diễn biến theo thời gian. Một người có thể cúi, ngồi hoặc nằm nghỉ với tư thế gần giống sau ngã, nhưng quá trình chuyển động trước đó lại khác nhau. Khi dùng chuỗi 48 frame đặc trưng, mô hình có thêm thông tin về hướng chuyển động, tốc độ rơi và sự thay đổi hình dạng cơ thể, từ đó hỗ trợ phân biệt giữa hành động sinh hoạt bình thường và sự cố té ngã.

## 2.6. State Machine Trong Phát Hiện Té Ngã

### 2.6.1. Lý Do Cần State Machine

Trong môi trường thực tế, mô hình học máy có thể dự đoán sai do dữ liệu thiếu, camera bị che khuất hoặc hành động người dùng gần giống té ngã. Vì vậy, đề tài bổ sung một state machine để xác nhận quyết định cuối cùng.

### 2.6.2. Các Tín Hiệu Được Kiểm Tra

State machine sử dụng các tín hiệu sau:

- Xác suất lớp FALLING và FALLEN từ mô hình GRU.
- Tốc độ rơi xuống của tâm cơ thể.
- Mức giảm chiều cao cơ thể so với trạng thái đứng trước đó.
- Tư thế nằm ngang của thân và toàn thân.
- Số frame liên tiếp có bằng chứng té ngã.
- Trạng thái đứng vững trước khi xảy ra sự kiện.
- Trạng thái nằm chậm, ngồi hoặc cúi để chặn báo nhầm.
- Thời gian nằm bất động sau ngã để nâng cấp cảnh báo FAINT.

### 2.6.3. Ý Nghĩa Giảm Báo Nhầm

Nhờ state machine, hệ thống có thể đưa ra quyết định ổn định hơn. Ví dụ, nếu người nằm xuống chậm và không có vận tốc rơi, hệ thống có thể phân loại là LYING_INTENTIONAL thay vì FALLEN. Nếu người đã ngã và nằm bất động trong thời gian dài, hệ thống có thể nâng cấp trạng thái thành FAINT.

## 2.7. Tracking Người Theo Thời Gian

### 2.7.1. Vai Trò Của Tracking

Khi trong khung hình có nhiều người hoặc khi detection bị mất ngắn hạn, hệ thống cần duy trì ID ổn định cho từng người. Nếu không có tracking, chuỗi đặc trưng có thể bị đứt đoạn và state machine không thể tích lũy bằng chứng chính xác.

### 2.7.2. Tiêu Chí Duy Trì Track

Module `PersonTracker` trong `phan_mem/src/tracker.py` sử dụng các tiêu chí:

- IoU giữa bounding box cũ và mới.
- Khoảng cách tâm box.
- Tỉ lệ giao nhau theo chiều ngang.
- Tỉ lệ diện tích.
- Smoothing bounding box để giảm nhiễu.
- Cơ chế giữ track khi mất detection trong một số frame.

### 2.7.3. Kết Quả Hỗ Trợ Cho Phát Hiện Té Ngã

Tracking giúp hệ thống biết trạng thái của từng người qua thời gian, từ đó phát hiện sự kiện té ngã theo từng track riêng.

## 2.8. Web Dashboard

### 2.8.1. Vai Trò Của Dashboard

Web dashboard là thành phần giúp người giám sát tương tác với hệ thống. Dashboard hiển thị luồng camera, trạng thái người trong từng camera, lịch sử cảnh báo, ảnh snapshot và danh sách video output.

### 2.8.2. Các Endpoint Chính

Trong project, web dashboard được cài đặt trong `phan_mem/web_server.py`. Server cung cấp các endpoint như:

**Bảng 2.4. Các endpoint chính của web dashboard**

| Endpoint | Chức năng |
|---|---|
| `/login` | Đăng nhập hệ thống |
| `/face_auth` | Xác thực khuôn mặt |
| `/dashboard` | Giao diện giám sát chính |
| `/status` | Trả về trạng thái camera/track dạng JSON |
| `/video_feed` | Truyền luồng MJPEG camera |
| `/alerts` | Trả về danh sách cảnh báo |
| `/videos` | Trả về danh sách video output |
| `/ack_alert` | Xác nhận cảnh báo an toàn |

### 2.8.3. Ý Nghĩa Vận Hành

Dashboard có ý nghĩa quan trọng vì nó biến kết quả xử lý của mô hình thành thông tin dễ theo dõi. Người giám sát không cần đọc log kỹ thuật mà có thể quan sát trực tiếp camera, trạng thái từng người và lịch sử cảnh báo. Đây cũng là nơi liên kết các thành phần khác như xác thực khuôn mặt, Telegram, video output và xác nhận an toàn sau cảnh báo.

## 2.9. Telegram Bot API

### 2.9.1. Vai Trò Cảnh Báo Từ Xa

Telegram Bot API cho phép hệ thống gửi tin nhắn và ảnh đến người dùng hoặc nhóm chat. Trong đề tài, Telegram được dùng làm kênh cảnh báo từ xa. Khi phát hiện té ngã, hệ thống gửi tin nhắn kèm ảnh snapshot, thông tin camera, track ID, trạng thái, độ tin cậy và link xác nhận an toàn.

### 2.9.2. Ý Nghĩa Đối Với Người Giám Sát

Cơ chế này giúp người thân hoặc nhân viên giám sát nhận cảnh báo ngay cả khi không mở dashboard.

## 2.10. ESP32 Và IoT

### 2.10.1. Vai Trò Của ESP32

ESP32 là vi điều khiển có WiFi tích hợp, phù hợp cho các ứng dụng IoT. Trong đề tài, ESP32 đóng vai trò thiết bị cảnh báo âm thanh tại chỗ. Thiết bị kết nối WiFi, gọi endpoint `/status` của web server, phân tích JSON và phát cảnh báo khi trạng thái nguy hiểm xuất hiện.

### 2.10.2. Ý Nghĩa Cảnh Báo Tại Chỗ

ESP32 sử dụng module âm thanh I2S/MAX98357A để xuất âm thanh ra loa. Việc tích hợp ESP32 giúp hệ thống không chỉ gửi cảnh báo từ xa mà còn phát cảnh báo trực tiếp trong khu vực lắp đặt camera.

## 2.11. Xác Thực Khuôn Mặt

### 2.11.1. Mục Đích Xác Thực

Xác thực khuôn mặt được sử dụng để tăng mức bảo vệ cho dashboard. Người dùng không chỉ nhập tài khoản/mật khẩu mà còn cần xác thực khuôn mặt bằng webcam trình duyệt. Ứng dụng `quetmat.py` dùng để đăng ký thông tin và encoding khuôn mặt người giám sát vào database.

### 2.11.2. Luồng Xử Lý Xác Thực

Trong web server, module `FaceAuthenticator` đọc database, trích xuất khuôn mặt từ ảnh base64 gửi lên và so khớp với dữ liệu đã đăng ký. Nếu khuôn mặt khớp, người dùng mới được vào dashboard.

## 2.12. DroidCam Trong Thu Nhận Camera Không Dây

### 2.12.1. Mục Đích Sử Dụng DroidCam

Trong quá trình thử nghiệm hệ thống, ngoài webcam gắn trực tiếp vào máy tính, đề tài còn sử dụng DroidCam để biến điện thoại thành camera không dây. DroidCam cho phép điện thoại truyền luồng video qua mạng WiFi, sau đó web server có thể đọc luồng này như một nguồn camera IP.

### 2.12.2. Ý Nghĩa Trong Thử Nghiệm Và Demo

Cách triển khai này có ý nghĩa thực tế vì không phải lúc nào hệ thống cũng có sẵn camera USB đặt đúng vị trí. Việc dùng điện thoại làm camera giúp quá trình demo linh hoạt hơn, có thể đặt camera ở nhiều góc quan sát khác nhau mà không cần kéo dây. Nguồn DroidCam thường có dạng URL nội bộ, ví dụ `http://<dia-chi-ip-dien-thoai>:4747/video`, sau đó được nhập vào dashboard hoặc cấu hình làm nguồn camera.

### 2.12.3. Vị Trí Trong Pipeline

Trong hệ thống, DroidCam đóng vai trò nguồn dữ liệu đầu vào. Sau khi frame được đọc bằng OpenCV, các bước xử lý phía sau vẫn giữ nguyên: YOLOv8 Pose phát hiện người, tracker gán ID, GRU phân loại trạng thái và state machine xác nhận té ngã.

## 2.13. Cloudflare Trong Truy Cập Tên Miền Và Public URL

### 2.13.1. Mục Đích Sử Dụng Cloudflare

Cloudflare được sử dụng để hỗ trợ truy cập hệ thống thông qua tên miền dễ nhớ thay vì phải dùng địa chỉ IP nội bộ hoặc IP động. Khi chạy demo hoặc cần gửi link xác nhận an toàn qua Telegram, một public URL ổn định giúp người giám sát có thể mở dashboard hoặc trang xác nhận từ thiết bị khác thuận tiện hơn.

### 2.13.2. Các Chức Năng Được Hỗ Trợ

Trong đề tài, Cloudflare có thể được dùng theo hướng tạo tunnel hoặc ánh xạ tên miền đến web server đang chạy local. Cách này giúp hạn chế việc phải mở port trực tiếp trên router, đồng thời tạo URL công khai cho các chức năng như:

- Truy cập dashboard từ bên ngoài mạng nội bộ.
- Tạo link xác nhận cảnh báo an toàn trong Telegram.
- Cung cấp endpoint `/status` ổn định cho thiết bị khác nếu cần.
- Giúp quá trình demo dễ sử dụng hơn nhờ tên miền ngắn và dễ nhớ.

### 2.13.3. Cấu Hình Public URL

Khi triển khai, biến `PUBLIC_BASE_URL` trong `.env` được dùng để cấu hình địa chỉ public của hệ thống. Đây là địa chỉ nền để hệ thống tạo link cảnh báo và link xác nhận gửi qua Telegram.

---

# CHƯƠNG 3. PHÂN TÍCH VÀ THIẾT KẾ HỆ THỐNG

## 3.1. Tổng Quan Hệ Thống

### 3.1.1. Các Thành Phần Chính

Hệ thống được thiết kế theo hướng xử lý liên tục từ camera đến cảnh báo. Camera cung cấp frame đầu vào. YOLOv8 Pose phát hiện người và keypoint. Tracker gán ID cho từng người. Module trích xuất đặc trưng tạo vector 37 chiều. Mô hình GRU phân loại trạng thái theo chuỗi thời gian. State machine xác nhận té ngã và giảm báo nhầm. Kết quả cuối cùng được đưa lên dashboard, gửi Telegram và truyền cho ESP32 phát cảnh báo âm thanh.

### 3.1.2. Luồng Xử Lý Tổng Quan

Sơ đồ tổng quan:

```text
Camera / Video
    |
    v
YOLOv8 Pose
    |
    v
Person Tracker
    |
    v
Feature Extraction
    |
    v
GRU Temporal Classifier
    |
    v
Fall State Machine
    |
    +------------------+------------------+------------------+
    |                  |                  |                  |
    v                  v                  v
Web Dashboard      Telegram Alert      ESP32 Audio Alert
```

## 3.2. Yêu Cầu Chức Năng

**Bảng 3.1. Danh sách yêu cầu chức năng của hệ thống**

| Mã | Yêu cầu chức năng |
|---|---|
| F1 | Nhận nguồn video từ webcam, camera IP hoặc file video |
| F2 | Phát hiện người và keypoint tư thế bằng YOLOv8 Pose |
| F3 | Theo dõi từng người bằng track ID |
| F4 | Trích xuất đặc trưng hình học, tư thế và chuyển động |
| F5 | Phân loại trạng thái bằng mô hình GRU |
| F6 | Xác nhận té ngã bằng state machine |
| F7 | Giảm báo nhầm khi ngồi, cúi, nằm chủ động hoặc đi sát camera |
| F8 | Hiển thị dashboard giám sát nhiều camera |
| F9 | Lưu ảnh cảnh báo, video output và lịch sử cảnh báo |
| F10 | Gửi cảnh báo Telegram kèm ảnh snapshot |
| F11 | Cho phép xác nhận an toàn để dừng nhắc lại |
| F12 | Xác thực người giám sát bằng tài khoản và khuôn mặt |
| F13 | ESP32 đọc trạng thái và phát cảnh báo âm thanh |
| F14 | Hỗ trợ nguồn camera không dây từ DroidCam hoặc camera IP |
| F15 | Hỗ trợ truy cập hệ thống qua tên miền/public URL bằng Cloudflare |

## 3.3. Yêu Cầu Phi Chức Năng

**Bảng 3.2. Danh sách yêu cầu phi chức năng của hệ thống**

| Nhóm | Yêu cầu |
|---|---|
| Thời gian thực | Xử lý camera với độ trễ thấp để cảnh báo kịp thời |
| Độ tin cậy | Hạn chế báo nhầm trong sinh hoạt thường ngày |
| Mở rộng | Hỗ trợ nhiều camera và cấu hình linh hoạt |
| Bảo mật | Có đăng nhập, xác thực khuôn mặt và token xác nhận cảnh báo |
| Dễ sử dụng | Dashboard trực quan, lịch sử cảnh báo rõ ràng |
| Dễ bảo trì | Code chia thành module độc lập |
| Tính thực tế | Có kênh cảnh báo từ xa và cảnh báo tại chỗ |

## 3.4. Thiết Kế Nhãn Trạng Thái

Hệ thống chuẩn hóa hành vi thành 8 lớp:

**Bảng 3.3. Bộ nhãn trạng thái sử dụng trong hệ thống**

| ID | Nhãn | Ý nghĩa |
|---:|---|---|
| 0 | NORMAL | Bình thường hoặc chưa đủ thông tin |
| 1 | WALKING_STANDING | Đi hoặc đứng |
| 2 | SITTING | Ngồi |
| 3 | BENDING | Cúi người |
| 4 | LYING_INTENTIONAL | Nằm chủ động |
| 5 | FALLING | Đang ngã |
| 6 | FALLEN | Đã ngã, nằm sau ngã |
| 7 | UNCONSCIOUS | Bất tỉnh hoặc bất động lâu |

Việc tách FALLING và FALLEN là cần thiết. FALLING thể hiện giai đoạn chuyển tiếp, có ý nghĩa cảnh báo sớm. FALLEN thể hiện trạng thái sau khi ngã, giúp xác nhận sự cố và theo dõi thời gian bất động.

## 3.5. Thiết Kế Module Phần Mềm

**Bảng 3.4. Các module phần mềm chính trong hệ thống**

| Thành phần | File chính | Vai trò |
|---|---|---|
| Cấu hình và tiện ích | `phan_mem/src/utils.py` | Load config, logging, path, video writer |
| Đọc dataset | `phan_mem/src/dataset_readers.py` | Đọc GMDCSA24, LE2I, UP-Fall |
| Chuẩn hóa nhãn | `phan_mem/src/label_mapping.py` | Map nhãn thô sang 8 lớp |
| Trích xuất đặc trưng | `phan_mem/src/features.py` | YOLO pose, keypoint, feature vector |
| Dataset huấn luyện | `phan_mem/src/dataset.py` | Tạo chuỗi feature cho GRU |
| Mô hình GRU | `phan_mem/src/models.py` | TemporalGRUFallClassifier |
| Huấn luyện | `phan_mem/train.py` | Train, validation, checkpoint |
| Suy luận | `phan_mem/inference.py` | Chạy webcam/video |
| Tracking | `phan_mem/src/tracker.py` | Theo dõi người theo track ID |
| Logic té ngã | `phan_mem/src/fall_logic.py` | State machine xác nhận té ngã |
| Cảnh báo | `phan_mem/src/alert_manager.py` | Telegram, snapshot, ack |
| Dashboard | `phan_mem/web_server.py` | Web server và giao diện |
| Xác thực khuôn mặt | `phan_mem/src/face_auth.py` | So khớp khuôn mặt |
| Quét mặt | `quetmat.py` | Tạo database khuôn mặt |
| ESP32 | `PhanCung/Esp32/src/main.cpp` | Cảnh báo âm thanh |

## 3.6. Luồng Xử Lý Camera

Luồng xử lý một frame trong hệ thống:

1. Đọc frame từ camera hoặc video.
2. Chạy YOLOv8 Pose để phát hiện người.
3. Lọc detection có độ tin cậy thấp hoặc bị trùng lặp.
4. Cập nhật tracker để gán ID cho từng người.
5. Tính vector đặc trưng 37 chiều cho từng track.
6. Cập nhật chuỗi đặc trưng theo thời gian.
7. Khi đủ chuỗi, đưa vào mô hình GRU để lấy xác suất 8 lớp.
8. Đưa xác suất và đặc trưng hiện tại vào state machine.
9. Nếu state machine xác nhận nguy hiểm, tạo cảnh báo.
10. Dashboard cập nhật trạng thái; Telegram gửi ảnh; ESP32 phát âm thanh.

## 3.7. Thiết Kế State Machine

### 3.7.1. Dữ Liệu Theo Dõi Cho Từng Track

State machine duy trì trạng thái riêng cho từng track. Mỗi track có các thông tin như:

- `status`: trạng thái hiện tại.
- `stable_upright_count`: số frame đứng vững trước đó.
- `fall_event_seen`: đã thấy sự kiện ngã hay chưa.
- `fall_event_time`: thời điểm bắt đầu sự kiện.
- `fallen_start_time`: thời điểm bắt đầu nằm sau ngã.
- `still_start_time`: thời điểm bắt đầu bất động.
- `velocity_history`: lịch sử bằng chứng rơi nhanh.
- `evidence_history`: lịch sử bằng chứng té ngã.
- `intentional_lying_start_time`: thời điểm bắt đầu nằm chủ động.

### 3.7.2. Các Trạng Thái Chính

Các trạng thái chính:

```text
NORMAL / WALKING_STANDING / SITTING / BENDING
        |
        | Có bằng chứng rơi nhanh + tư thế phù hợp + xác suất mô hình
        v
FALLING
        |
        | Nằm sau ngã đủ thời gian
        v
FALLEN
        |
        | Bất động quá thời gian cấu hình
        v
FAINT / UNCONSCIOUS
```

### 3.7.3. Cơ Chế Phục Hồi Và Giảm Báo Nhầm

Nếu người phục hồi về tư thế đứng/ngồi trong thời gian cho phép, state machine đưa trạng thái trở lại bình thường. Nếu người nằm xuống chậm và không có bằng chứng rơi, hệ thống ưu tiên trạng thái LYING_INTENTIONAL.

## 3.8. Thiết Kế Cảnh Báo

### 3.8.1. Cấu Trúc Dữ Liệu Cảnh Báo

Mỗi cảnh báo gồm các trường:

**Bảng 3.5. Các trường dữ liệu của một cảnh báo**

| Trường | Ý nghĩa |
|---|---|
| alert_id | Mã cảnh báo duy nhất |
| camera_id | Camera phát hiện sự cố |
| track_id | ID người trong camera |
| status | Trạng thái nguy hiểm |
| confidence | Độ tin cậy mô hình |
| fall_score | Điểm té ngã từ state machine |
| reason | Lý do ra quyết định |
| timestamp_sec | Thời điểm trong video |
| image_path | Ảnh snapshot cảnh báo |
| acknowledged | Đã xác nhận an toàn hay chưa |

### 3.8.2. Điều Kiện Gửi Và Nhắc Lại Cảnh Báo

Các trạng thái nguy hiểm gồm FALLING, FALLEN, FAINT và UNCONSCIOUS. Cảnh báo được lọc theo số frame hoặc thời gian tối thiểu để tránh gửi cảnh báo do nhiễu ngắn. Hệ thống cũng có cooldown để tránh gửi liên tục nhiều tin nhắn trùng lặp.

## 3.9. Thiết Kế Xác Thực

Luồng xác thực dashboard:

```text
Người dùng mở web
    |
    v
Đăng nhập tài khoản/mật khẩu
    |
    v
Mở trang quét mặt
    |
    v
Gửi ảnh khuôn mặt lên server
    |
    v
So khớp với database khuôn mặt
    |
    v
Cho phép vào dashboard nếu hợp lệ
```

Cách xác thực hai bước giúp dashboard tránh bị truy cập bởi người không được cấp quyền, nhất là khi hệ thống có thể hiển thị hình ảnh camera và lịch sử cảnh báo.

## 3.10. Thiết Kế ESP32 Cảnh Báo Âm Thanh

ESP32 hoạt động theo chu kỳ:

1. Kết nối WiFi.
2. Gọi HTTP GET đến endpoint `/status`.
3. Parse JSON bằng ArduinoJson.
4. Kiểm tra trạng thái của từng camera/track.
5. Nếu phát hiện FALLEN hoặc FAINT, đưa sự kiện vào hàng đợi âm thanh.
6. Phát cảnh báo tiếng Việt qua module I2S.

Thiết kế hàng đợi âm thanh giúp ESP32 không bị nghẽn khi có nhiều sự kiện liên tiếp. Thiết bị cũng hỗ trợ nhận cảnh báo qua Serial để kiểm thử nhanh khi chưa kết nối web server.

## 3.11. Thiết Kế Nguồn Camera Không Dây Bằng DroidCam

Trong hệ thống, nguồn camera không bị giới hạn ở webcam USB. Dashboard và web server cho phép nhập nguồn camera dưới dạng URL, nhờ đó có thể sử dụng DroidCam trên điện thoại để tạo camera không dây. Điện thoại và máy tính chạy web server cùng kết nối vào một mạng WiFi, DroidCam phát luồng video qua địa chỉ nội bộ, sau đó OpenCV đọc luồng này để xử lý.

Luồng dữ liệu với DroidCam:

```text
Điện thoại Android
    |
    | DroidCam phát video qua WiFi
    v
URL camera nội bộ
    |
    | OpenCV đọc frame
    v
Web server / InferenceStream
    |
    v
Pipeline phát hiện té ngã
```

Việc dùng DroidCam giúp tăng tính linh hoạt khi demo. Người thực hiện có thể đặt điện thoại tại vị trí phù hợp để ghi lại toàn thân người dùng, trong khi máy tính vẫn chạy mô hình AI và dashboard.

## 3.12. Thiết Kế Truy Cập Tên Miền Bằng Cloudflare

Cloudflare được đặt ở lớp truy cập bên ngoài của hệ thống. Web server vẫn chạy trên máy tính local, nhưng thông qua Cloudflare, người dùng có thể truy cập bằng tên miền dễ nhớ. Điều này đặc biệt hữu ích với Telegram, vì link xác nhận an toàn cần là đường dẫn mà người nhận có thể mở từ điện thoại.

Sơ đồ truy cập qua Cloudflare:

```text
Người giám sát / Telegram
        |
        v
Tên miền Cloudflare
        |
        v
Cloudflare Tunnel / DNS
        |
        v
Web server local
        |
        +--> Dashboard
        +--> /alerts
        +--> /ack_alert
        +--> /status
```

Trong triển khai, `PUBLIC_BASE_URL` được cấu hình bằng tên miền Cloudflare. Khi Telegram gửi cảnh báo, link xác nhận an toàn sẽ dùng địa chỉ này để người giám sát có thể truy cập thuận tiện hơn.

## 3.13. Sơ Đồ Khối Tổng Thể Hệ Thống

Sơ đồ khối tổng thể giúp mô tả toàn bộ hệ thống ở mức kiến trúc. Đây là sơ đồ nên đặt trong chương phân tích và thiết kế để hội đồng dễ hình dung các thành phần chính.

```text
+--------------------+        +----------------------+
| Webcam / DroidCam  |        | Video local/IP camera|
+---------+----------+        +----------+-----------+
          |                              |
          +--------------+---------------+
                         |
                         v
              +---------------------+
              |  OpenCV đọc frame   |
              +----------+----------+
                         |
                         v
              +---------------------+
              |   YOLOv8 Pose       |
              | Người + keypoint    |
              +----------+----------+
                         |
                         v
              +---------------------+
              |  Person Tracker     |
              |  Gán track ID       |
              +----------+----------+
                         |
                         v
              +---------------------+
              | Trích xuất feature  |
              | 37 đặc trưng/frame  |
              +----------+----------+
                         |
                         v
              +---------------------+
              |  GRU Classifier     |
              | Phân loại trạng thái|
              +----------+----------+
                         |
                         v
              +---------------------+
              | Fall State Machine  |
              | Xác nhận té ngã     |
              +----------+----------+
                         |
       +-----------------+------------------+
       |                 |                  |
       v                 v                  v
+-------------+   +---------------+   +----------------+
| Dashboard   |   | Telegram Bot  |   | ESP32 + Loa    |
| Web server  |   | Ảnh + link ack|   | Cảnh báo âm thanh|
+------+------+   +-------+-------+   +----------------+
       |
       v
+----------------+
| Cloudflare URL |
| Tên miền dễ mở |
+----------------+
```

## 3.14. Lưu Đồ Thuật Toán Phát Hiện Té Ngã

Lưu đồ thuật toán mô tả logic xử lý chính của hệ thống trong quá trình chạy camera.

```text
Bắt đầu
   |
   v
Khởi tạo config, model YOLOv8 Pose, GRU, tracker, state machine
   |
   v
Mở nguồn camera/video
   |
   v
Đọc frame mới
   |
   +-- Không đọc được frame? -- Có --> Kết thúc hoặc thử kết nối lại
   |
  Không
   |
   v
Phát hiện người và keypoint bằng YOLOv8 Pose
   |
   v
Cập nhật tracker và gán track ID
   |
   v
Trích xuất vector đặc trưng 37 chiều cho từng track
   |
   v
Cập nhật chuỗi đặc trưng theo thời gian
   |
   +-- Chưa đủ sequence_length? -- Có --> Hiển thị trạng thái tạm thời và đọc frame tiếp
   |
  Không
   |
   v
Đưa chuỗi đặc trưng vào mô hình GRU
   |
   v
Nhận xác suất các trạng thái
   |
   v
State machine kiểm tra bằng chứng té ngã
   |
   +-- Có FALLING/FALLEN/FAINT? -- Không --> Cập nhật dashboard và đọc frame tiếp
   |
  Có
   |
   v
Tạo cảnh báo và snapshot
   |
   v
Gửi Telegram, cập nhật dashboard, ESP32 phát âm thanh
   |
   v
Tiếp tục theo dõi
```

## 3.15. Lưu Đồ Thuật Toán Cảnh Báo Và Xác Nhận An Toàn

Ngoài thuật toán phát hiện té ngã, hệ thống còn có luồng cảnh báo riêng để đảm bảo người giám sát nhận được thông tin và có thể xác nhận tình trạng an toàn.

```text
State machine phát hiện trạng thái nguy hiểm
   |
   v
AlertManager kiểm tra bộ lọc báo nhầm
   |
   +-- Chưa đủ số frame/thời gian nguy hiểm? -- Có --> Không gửi cảnh báo
   |
  Không
   |
   v
Tạo alert_id và ảnh snapshot
   |
   v
Lưu bản ghi cảnh báo vào bộ nhớ/lịch sử
   |
   v
Gửi tin nhắn Telegram kèm ảnh và link xác nhận
   |
   v
ESP32 đọc /status và phát cảnh báo âm thanh
   |
   v
Người giám sát bấm link xác nhận an toàn?
   |
   +-- Có --> Đánh dấu acknowledged, dừng nhắc lại
   |
   +-- Không --> Nhắc lại Telegram theo cấu hình
```

---

# CHƯƠNG 4. XÂY DỰNG VÀ CÀI ĐẶT HỆ THỐNG

## 4.1. Môi Trường Phát Triển

**Bảng 4.1. Môi trường và công nghệ phát triển hệ thống**

| Thành phần | Công nghệ |
|---|---|
| Ngôn ngữ chính | Python |
| Deep learning | PyTorch |
| Pose estimation | Ultralytics YOLOv8 Pose |
| Xử lý ảnh/video | OpenCV |
| Xử lý dữ liệu | NumPy, Pandas, scikit-learn |
| Web server | Python ThreadingHTTPServer |
| Giao diện | HTML, CSS, JavaScript |
| Xác thực khuôn mặt | face_recognition |
| Cảnh báo | Telegram Bot API |
| Camera không dây | DroidCam |
| Tên miền/public URL | Cloudflare |
| Phần cứng | ESP32, I2S/MAX98357A |

Các thư viện chính trong `requirements.txt`:

- torch
- torchvision
- ultralytics
- opencv-python
- numpy
- pandas
- PyYAML
- tqdm
- requests
- scikit-learn
- scipy
- matplotlib
- gdown

## 4.2. Cấu Trúc Project

```text
c:\DATN
|-- quetmat.py
|-- yolov8n-pose.pt
|-- data
|   |-- database.json
|   |-- faces
|-- PhanCung
|   |-- Esp32
|       |-- src
|           |-- main.cpp
|-- phan_mem
    |-- configs
    |   |-- config.yaml
    |-- src
    |   |-- alert_manager.py
    |   |-- dataset.py
    |   |-- dataset_readers.py
    |   |-- face_auth.py
    |   |-- fall_logic.py
    |   |-- features.py
    |   |-- label_mapping.py
    |   |-- metrics.py
    |   |-- models.py
    |   |-- tracker.py
    |   |-- utils.py
    |-- train.py
    |-- inference.py
    |-- web_server.py
    |-- prepare_dataset.py
    |-- download_datasets.py
    |-- checkpoints
    |   |-- best_model.pt
    |   |-- last_model.pt
    |-- data
    |   |-- manifests
    |   |-- raw
    |   |-- cache
    |-- outputs
    |   |-- alerts
    |   |-- fall_clips
    |   |-- person_clips
    |   |-- events
    |-- logs
```

## 4.3. Cấu Hình Hệ Thống

### 4.3.1. File Cấu Hình Chính

File cấu hình chính là `phan_mem/configs/config.yaml`.

### 4.3.2. Các Tham Số Quan Trọng

Một số cấu hình quan trọng:

**Bảng 4.2. Các tham số cấu hình chính của hệ thống**

| Nhóm | Tham số | Giá trị |
|---|---|---:|
| Detector | model | yolov8n-pose.pt |
| Detector | confidence | 0.38 |
| Detector | image_size | 640 |
| Feature | sample_fps | 10.0 |
| Feature | sequence_length | 48 |
| Feature | window_stride | 2 |
| Model | input_dim | 37 |
| Model | hidden_dim | 128 |
| Model | num_layers | 2 |
| Model | bidirectional | true |
| Model | num_classes | 8 |
| Train | epochs | 60 |
| Train | batch_size | 32 |
| Train | learning_rate | 0.001 |
| Train | patience | 12 |
| Web | camera_count | 4 |

Các thông tin nhạy cảm như token Telegram, mật khẩu đăng nhập, secret xác nhận và đường dẫn database khuôn mặt được đặt trong `.env`, không nên đưa vào báo cáo công khai.

## 4.4. Chuẩn Bị Dữ Liệu

### 4.4.1. Nguồn Dữ Liệu

Hệ thống sử dụng dữ liệu từ nhiều nguồn, trong đó có GMDCSA24, LE2I và UP-Fall. Dữ liệu được chuẩn hóa thành manifest thống nhất.

### 4.4.2. Thống Kê Dữ Liệu

Thống kê hiện tại:

**Bảng 4.3. Thống kê số mẫu hợp lệ theo từng bộ dữ liệu**

| Dataset | Số mẫu hợp lệ |
|---|---:|
| GMDCSA24 | 160 |
| LE2I | 131 |
| UP-Fall | 2 |
| Tổng | 293 |

Chia tập:

**Bảng 4.4. Tỉ lệ chia tập dữ liệu huấn luyện, validation và test**

| Split | Số mẫu |
|---|---:|
| Train | 259 |
| Validation | 32 |
| Test | 2 |

Phân bố nhãn toàn bộ:

**Bảng 4.5. Phân bố số lượng mẫu theo từng nhãn trạng thái**

| Nhãn | Số lượng |
|---|---:|
| NORMAL | 58 |
| WALKING_STANDING | 260 |
| SITTING | 78 |
| BENDING | 0 |
| LYING_INTENTIONAL | 15 |
| FALLING | 211 |
| FALLEN | 218 |
| UNCONSCIOUS | 0 |

### 4.4.3. Nhận Xét Về Dữ Liệu

Nhận xét: tập dữ liệu hiện tại có nhiều mẫu FALLING và FALLEN, phù hợp mục tiêu chính là phát hiện té ngã. Tuy nhiên, dữ liệu chưa cân bằng; một số lớp như BENDING và UNCONSCIOUS chưa có mẫu, do đó chưa thể đánh giá đầy đủ các lớp này bằng validation.

## 4.5. Trích Xuất Đặc Trưng

Module `features.py` thực hiện các bước:

1. Đọc frame video theo tần suất lấy mẫu.
2. Chạy YOLOv8 Pose.
3. Lấy bounding box, confidence và keypoint.
4. Chọn hoặc lọc người phù hợp.
5. Tính đặc trưng hình học, tư thế và chuyển động.
6. Gán nhãn cho từng timestamp theo annotation.
7. Lưu cache feature vào `data/cache/features`.

Vector đặc trưng 37 chiều giúp mô hình tập trung vào hình dạng và chuyển động cơ thể, thay vì phải học trực tiếp từ ảnh thô. Điều này phù hợp với project vì mục tiêu là xây dựng hệ thống chạy được trên camera thật và có tốc độ xử lý chấp nhận được.

## 4.6. Huấn Luyện Mô Hình

File `train.py` xây dựng quy trình huấn luyện:

- Load cấu hình.
- Khởi tạo YOLOv8 Pose detector.
- Tạo dataset từ train/val manifest.
- Khởi tạo mô hình GRU.
- Tính class weight để xử lý mất cân bằng dữ liệu.
- Huấn luyện bằng CrossEntropyLoss.
- Tối ưu bằng AdamW.
- Dùng ReduceLROnPlateau để giảm learning rate khi metric không cải thiện.
- Lưu checkpoint tốt nhất theo `fall_safety_score`.
- Dừng sớm nếu không cải thiện sau `patience = 12` epoch.

Chỉ số `fall_safety_score` được thiết kế để ưu tiên an toàn:

```text
fall_safety_score =
0.45 * recall_falling
+ 0.25 * f1_falling
+ 0.20 * f1_fallen
+ 0.10 * f1_macro
```

Điều này phù hợp với bài toán vì trong phát hiện té ngã, bỏ sót sự kiện nguy hiểm thường nghiêm trọng hơn một số báo nhầm có thể kiểm soát bằng xác nhận.

## 4.7. Xây Dựng Dashboard

Dashboard trong `web_server.py` có các chức năng:

- Đăng nhập hệ thống.
- Xác thực khuôn mặt.
- Hiển thị nhiều camera.
- Hiển thị số người, FPS, nguồn camera và thời gian.
- Hiển thị trạng thái từng track.
- Xem lịch sử cảnh báo.
- Xem ảnh snapshot cảnh báo.
- Xem video output đã lưu.
- Cung cấp endpoint `/status` cho ESP32.

Dashboard được thiết kế để phục vụ luồng demo thực tế: người giám sát đăng nhập, xác thực khuôn mặt, mở camera, quan sát trạng thái, nhận cảnh báo và xác nhận an toàn.

## 4.8. Tích Hợp DroidCam Làm Camera Không Dây

Trong quá trình demo, DroidCam được sử dụng để biến điện thoại thành camera không dây. Sau khi điện thoại và máy tính cùng kết nối vào mạng WiFi, ứng dụng DroidCam cung cấp địa chỉ video nội bộ. Địa chỉ này được nhập vào hệ thống như một nguồn camera.

Ví dụ dạng nguồn camera:

```text
http://<dia-chi-ip-dien-thoai>:4747/video
```

Web server đọc nguồn này bằng OpenCV giống như camera IP. Sau khi đọc được frame, toàn bộ pipeline xử lý phía sau không thay đổi. Điều này giúp hệ thống có thể demo linh hoạt hơn, đặt camera ở vị trí quan sát toàn thân người dùng mà không phụ thuộc vào dây USB.

Trong báo cáo và khi bảo vệ, DroidCam nên được trình bày là giải pháp camera không dây phục vụ thử nghiệm và demo hệ thống.

## 4.9. Tích Hợp Cloudflare Để Truy Cập Bằng Tên Miền

Cloudflare được sử dụng để tạo tên miền hoặc public URL cho web server. Khi hệ thống chạy local, việc truy cập bằng địa chỉ IP nội bộ có thể khó nhớ và không thuận tiện khi gửi link qua Telegram. Cloudflare giúp ánh xạ một tên miền dễ dùng đến web server, từ đó người giám sát có thể mở dashboard hoặc link xác nhận an toàn dễ dàng hơn.

Trong hệ thống, public URL được cấu hình thông qua biến:

```text
PUBLIC_BASE_URL=
```

Khi biến này được cấu hình, Telegram có thể gửi link xác nhận dạng:

```text
https://<ten-mien>/ack_alert?id=<alert_id>&token=<token>
```

Nhờ đó, người nhận cảnh báo chỉ cần bấm vào link trong Telegram để xác nhận an toàn, thay vì phải nhập địa chỉ IP hoặc truy cập thủ công vào server.

## 4.10. Xây Dựng Cảnh Báo Telegram

Module `alert_manager.py` chịu trách nhiệm:

- Kiểm tra trạng thái nguy hiểm.
- Lọc cảnh báo ngắn để giảm báo nhầm.
- Tạo ảnh snapshot có khung cảnh báo.
- Lưu bản ghi cảnh báo.
- Gửi Telegram nếu cấu hình cho phép.
- Tạo link xác nhận an toàn.
- Nhắc lại cảnh báo nếu chưa được xác nhận.

Khi phát hiện nguy hiểm, người giám sát nhận được thông tin trực quan thay vì chỉ nhận dòng chữ. Ảnh snapshot giúp người nhận nhanh chóng đánh giá tình huống và quyết định hành động.

## 4.11. Xây Dựng Xác Thực Khuôn Mặt

Ứng dụng `quetmat.py` dùng để tạo database khuôn mặt nhân viên giám sát. Ứng dụng có giao diện desktop, cho phép nhập thông tin người giám sát và quét khuôn mặt bằng webcam. Dữ liệu được lưu vào `data/database.json`, ảnh khuôn mặt lưu trong `data/faces`.

Trong web server, người dùng sau khi đăng nhập phải quét mặt. Nếu khuôn mặt khớp với database và người đó đang active, hệ thống mới cho phép vào dashboard.

## 4.12. Xây Dựng ESP32

### 4.12.1. Chức Năng Chính Của ESP32

ESP32 trong `PhanCung/Esp32/src/main.cpp` có các chức năng:

- Kết nối WiFi.
- Gọi endpoint trạng thái của web server.
- Parse JSON trạng thái nhiều camera.
- Nhận biết FALLEN, FAINT hoặc UNCONSCIOUS.
- Đưa sự kiện vào hàng đợi âm thanh.
- Phát cảnh báo tiếng Việt qua TTS.
- Hỗ trợ test qua Serial.

### 4.12.2. Kết Nối Module Âm Thanh

Kết nối module âm thanh:

**Bảng 4.6. Kết nối GPIO giữa ESP32 và module âm thanh I2S/MAX98357A**

| Tín hiệu | GPIO |
|---|---:|
| BCLK | 26 |
| LRC/WS | 25 |
| DIN | 22 |

### 4.12.3. Lưu Ý Khi Triển Khai

Khi triển khai, các thông tin như WiFi SSID, mật khẩu và URL web server nên được cấu hình riêng, tránh đưa trực tiếp vào mã nguồn khi nộp hoặc chia sẻ.

---

# CHƯƠNG 5. THỰC NGHIỆM VÀ ĐÁNH GIÁ

## 5.1. Mục Tiêu Đánh Giá

Thực nghiệm nhằm đánh giá:

- Khả năng mô hình GRU phân loại trạng thái trên validation.
- Khả năng nhận diện giai đoạn FALLING và FALLEN.
- Khả năng giảm báo nhầm của state machine.
- Khả năng hoạt động của dashboard, Telegram và ESP32.
- Hạn chế hiện tại của dữ liệu và hệ thống.

## 5.2. Kết Quả Huấn Luyện

### 5.2.1. Kết Quả Tổng Quan

Kết quả tốt nhất được ghi nhận tại epoch 19.

**Bảng 5.1. Kết quả huấn luyện tổng quan của mô hình GRU**

| Chỉ số | Giá trị |
|---|---:|
| Epoch tốt nhất | 19 |
| Train loss | 0.221 |
| Validation loss | 1.923 |
| Accuracy | 0.600 |
| Precision macro | 0.395 |
| Recall macro | 0.405 |
| F1 macro | 0.383 |
| Precision weighted | 0.601 |
| Recall weighted | 0.600 |
| F1 weighted | 0.570 |
| Fall safety score | 0.761 |

### 5.2.2. Kết Quả Theo Từng Lớp

Kết quả theo từng lớp:

**Bảng 5.2. Chỉ số phân loại theo từng lớp trạng thái**

| Lớp | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| NORMAL | 0.459 | 0.736 | 0.565 | 91 |
| WALKING_STANDING | 0.529 | 0.429 | 0.474 | 21 |
| SITTING | 0.927 | 0.487 | 0.639 | 78 |
| BENDING | 0.000 | 0.000 | 0.000 | 0 |
| LYING_INTENTIONAL | 0.000 | 0.000 | 0.000 | 31 |
| FALLING | 0.765 | 0.897 | 0.825 | 58 |
| FALLEN | 0.478 | 0.688 | 0.564 | 16 |
| UNCONSCIOUS | 0.000 | 0.000 | 0.000 | 0 |

## 5.3. Nhận Xét Kết Quả

### 5.3.1. Kết Quả Đối Với Lớp Té Ngã

Lớp FALLING đạt F1-score 0.825 và recall 0.897. Đây là kết quả quan trọng vì FALLING là giai đoạn hệ thống cần phát hiện sớm để gửi cảnh báo. Recall cao cho thấy mô hình có xu hướng ít bỏ sót các mẫu đang ngã trong tập validation.

Lớp FALLEN đạt F1-score 0.564. Kết quả này thấp hơn FALLING vì trạng thái sau ngã có thể gần giống người nằm chủ động. Đây cũng là lý do đề tài cần state machine để xét thêm quá trình trước đó, thay vì chỉ dựa vào hình ảnh nằm ngang.

### 5.3.2. Kết Quả Đối Với Các Trạng Thái Sinh Hoạt

Lớp SITTING có precision cao 0.927 nhưng recall 0.487. Điều này cho thấy khi mô hình dự đoán ngồi thì khá chắc chắn, nhưng vẫn có một phần mẫu ngồi bị nhầm sang lớp khác.

Lớp LYING_INTENTIONAL chưa đạt kết quả tốt. Đây là điểm cần cải thiện vì nằm chủ động là tình huống rất dễ gây báo nhầm với té ngã. Trong hệ thống hiện tại, state machine hỗ trợ xử lý bằng cách kiểm tra tốc độ rơi, bằng chứng trước đó và quá trình nằm chậm.

### 5.3.3. Hạn Chế Dữ Liệu Khi Đánh Giá

BENDING và UNCONSCIOUS chưa có support trong validation nên chưa thể đánh giá khách quan. Đây là hạn chế dữ liệu cần bổ sung trong tương lai.

## 5.4. Ma Trận Nhầm Lẫn

Thứ tự nhãn: NORMAL, WALKING_STANDING, SITTING, BENDING, LYING_INTENTIONAL, FALLING, FALLEN, UNCONSCIOUS.

**Bảng 5.3. Ma trận nhầm lẫn của mô hình trên tập validation**

| True \ Pred | NORMAL | WALKING | SITTING | BENDING | LYING | FALLING | FALLEN | UNCONSCIOUS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NORMAL | 67 | 8 | 3 | 0 | 0 | 10 | 3 | 0 |
| WALKING_STANDING | 12 | 9 | 0 | 0 | 0 | 0 | 0 | 0 |
| SITTING | 40 | 0 | 38 | 0 | 0 | 0 | 0 | 0 |
| BENDING | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| LYING_INTENTIONAL | 26 | 0 | 0 | 0 | 0 | 2 | 3 | 0 |
| FALLING | 0 | 0 | 0 | 0 | 0 | 52 | 6 | 0 |
| FALLEN | 1 | 0 | 0 | 0 | 0 | 4 | 11 | 0 |
| UNCONSCIOUS | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## 5.5. Kiểm Thử Theo Kịch Bản Demo

Để đánh giá khả năng hoạt động của hệ thống ngoài validation, cần kiểm thử các tình huống thực tế bằng webcam hoặc camera IP.

**Bảng 5.4. Kịch bản kiểm thử demo hệ thống**

| Tình huống | Số lần test | Đúng | Sai | Kết quả mong muốn |
|---|---:|---:|---:|---|
| Đi/đứng bình thường | 10 |  |  | Không cảnh báo |
| Ngồi xuống | 10 |  |  | Không cảnh báo té ngã |
| Cúi nhặt đồ | 10 |  |  | Không cảnh báo té ngã |
| Nằm xuống chủ động | 10 |  |  | Không cảnh báo té ngã |
| Té ngã mô phỏng | 10 |  |  | Có cảnh báo |
| Nằm bất động sau ngã | 5 |  |  | Có thể nâng cấp FAINT |
| Mất track ngắn | 5 |  |  | Không báo nhầm liên tục |
| Telegram nhận cảnh báo | 5 |  |  | Có tin nhắn kèm ảnh |
| ESP32 phát âm thanh | 5 |  |  | Có cảnh báo giọng nói |

Phần này nên được điền sau khi thực hiện demo thật để báo cáo có số liệu thực nghiệm trực tiếp.

## 5.6. Kịch Bản Bảo Vệ Đề Xuất

Khi bảo vệ, nên demo theo trình tự:

1. Mở `quetmat.py` để giới thiệu database khuôn mặt người giám sát.
2. Chạy `web_server.py`.
3. Truy cập trang đăng nhập.
4. Đăng nhập bằng tài khoản.
5. Xác thực khuôn mặt.
6. Vào dashboard.
7. Mở camera hoặc nguồn video demo.
8. Thực hiện hành động bình thường: đi, đứng, ngồi.
9. Thực hiện té ngã mô phỏng.
10. Quan sát trạng thái trên dashboard chuyển sang FALLING/FALLEN.
11. Kiểm tra Telegram nhận cảnh báo kèm ảnh.
12. Kiểm tra ESP32 phát âm thanh cảnh báo.
13. Bấm link xác nhận an toàn để dừng nhắc lại.

## 5.7. Đánh Giá Chung

Hệ thống đã đạt được mục tiêu xây dựng một pipeline phát hiện té ngã tương đối hoàn chỉnh. Không chỉ có phần nhận diện bằng AI, hệ thống còn có giao diện giám sát, xác thực, lưu cảnh báo, gửi Telegram và cảnh báo âm thanh qua ESP32. Đây là điểm mạnh quan trọng khi trình bày đồ án, vì đề tài thể hiện được quá trình biến một mô hình nhận diện thành một hệ thống ứng dụng.

Kết quả validation cho thấy mô hình nhận diện tốt giai đoạn FALLING. Tuy nhiên, độ chính xác tổng thể chưa quá cao do dữ liệu còn hạn chế và mất cân bằng. Trong thực tế, state machine đóng vai trò bổ sung để giảm báo nhầm và làm cho cảnh báo ổn định hơn.

## 5.8. Hạn Chế

Các hạn chế hiện tại:

- Tập test trong manifest chỉ có 2 mẫu, chưa đủ để đánh giá tổng quát.
- Một số lớp thiếu dữ liệu, đặc biệt là BENDING và UNCONSCIOUS.
- Lớp LYING_INTENTIONAL còn khó phân biệt với FALLEN.
- Hệ thống phụ thuộc vào chất lượng camera, ánh sáng và góc nhìn.
- Khi cơ thể bị che khuất nhiều, keypoint có thể sai hoặc thiếu.
- Xác thực khuôn mặt chưa có chống giả mạo bằng ảnh/video.
- ESP32 dùng TTS online nên cần kết nối mạng ổn định.

---

# CHƯƠNG 6. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

## 6.1. Kết Luận

Đề tài đã xây dựng thành công hệ thống phát hiện té ngã và cảnh báo khẩn cấp cho người cao tuổi bằng camera, trí tuệ nhân tạo và IoT. Hệ thống sử dụng YOLOv8 Pose pretrained để phát hiện người và trích xuất keypoint, sau đó xây dựng vector đặc trưng 37 chiều mô tả hình học, tư thế và chuyển động. Mô hình GRU tự huấn luyện được sử dụng để phân loại trạng thái theo chuỗi thời gian.

Để tăng độ tin cậy trong môi trường thực tế, đề tài xây dựng thêm state machine nhằm xác nhận sự kiện té ngã dựa trên nhiều bằng chứng như tốc độ rơi, giảm chiều cao cơ thể, tư thế nằm ngang, thời gian nằm sau ngã và trạng thái bất động. Nhờ đó, hệ thống có thể giảm báo nhầm trong các tình huống như ngồi, cúi, nằm chủ động hoặc đi sát camera.

Ngoài phần AI, đề tài đã tích hợp các thành phần phục vụ vận hành thực tế: web dashboard giám sát nhiều camera, đăng nhập và xác thực khuôn mặt, cảnh báo Telegram kèm ảnh snapshot, link xác nhận an toàn và ESP32 phát cảnh báo âm thanh tại chỗ. Đây là một hệ thống AI + IoT hoàn chỉnh, có thể trình diễn theo luồng từ camera phát hiện sự cố đến người giám sát nhận cảnh báo và xác nhận an toàn.

Kết quả huấn luyện tốt nhất đạt accuracy validation 60%, F1 lớp FALLING 0.825 và fall_safety_score 0.761. Kết quả này cho thấy hướng tiếp cận có tiềm năng, đặc biệt ở việc nhận diện giai đoạn đang ngã. Tuy nhiên, hệ thống cần được mở rộng dữ liệu và kiểm thử thêm trong nhiều điều kiện thực tế để nâng cao độ ổn định.

## 6.2. Đóng Góp Của Đề Tài

Các đóng góp chính:

- Xây dựng pipeline phát hiện té ngã dựa trên pose estimation.
- Thiết kế bộ đặc trưng 37 chiều phù hợp với tư thế và chuyển động cơ thể.
- Huấn luyện mô hình GRU cho bài toán phân loại trạng thái té ngã.
- Xây dựng state machine giảm báo nhầm.
- Phát triển dashboard web giám sát nhiều camera.
- Tích hợp xác thực khuôn mặt cho người giám sát.
- Tích hợp cảnh báo Telegram kèm ảnh snapshot và xác nhận an toàn.
- Tích hợp ESP32 để phát cảnh báo âm thanh tại chỗ.
- Tổ chức project theo các module rõ ràng, thuận tiện bảo trì và mở rộng.

## 6.3. Hướng Phát Triển

Trong tương lai, hệ thống có thể được phát triển theo các hướng:

- Thu thập thêm dữ liệu thực tế từ nhiều góc camera và môi trường khác nhau.
- Bổ sung dữ liệu cho các lớp BENDING, LYING_INTENTIONAL và UNCONSCIOUS.
- Xây dựng tập test độc lập lớn hơn để đánh giá khách quan.
- So sánh GRU với LSTM, Temporal CNN hoặc Transformer nhẹ.
- Tối ưu tốc độ inference để chạy ổn định hơn trên máy cấu hình thấp.
- Thêm liveness detection cho xác thực khuôn mặt.
- Thêm phân quyền người dùng trên dashboard.
- Đóng gói web server thành service để triển khai lâu dài.
- Tích hợp thêm SMS, Zalo OA hoặc cuộc gọi tự động.
- Nâng cấp ESP32 phát âm thanh offline để giảm phụ thuộc vào TTS online.

---

# TÀI LIỆU THAM KHẢO

1. Ultralytics YOLOv8 Documentation, tài liệu YOLOv8 Pose.
2. PyTorch Documentation, tài liệu GRU, DataLoader và huấn luyện mô hình học sâu.
3. OpenCV Documentation, tài liệu xử lý ảnh và video.
4. scikit-learn Documentation, tài liệu precision, recall, F1-score và confusion matrix.
5. Telegram Bot API Documentation, tài liệu gửi tin nhắn, ảnh và xử lý bot.
6. Espressif ESP32 Documentation, tài liệu WiFi và lập trình ESP32.
7. ArduinoJson Documentation, tài liệu phân tích JSON trên vi điều khiển.
8. Tài liệu về các bộ dữ liệu phát hiện té ngã: GMDCSA24, LE2I và UP-Fall.

---

# PHỤ LỤC A. LỆNH CÀI ĐẶT VÀ CHẠY HỆ THỐNG

## A.1. Cài Đặt Môi Trường

```bash
cd c:\DATN\phan_mem
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Nếu dùng xác thực khuôn mặt, cần cài thêm `face_recognition` theo môi trường Windows phù hợp.

## A.2. Chuẩn Bị Dữ Liệu

```bash
cd c:\DATN\phan_mem
python download_datasets.py --config configs/config.yaml
python prepare_dataset.py --config configs/config.yaml
```

## A.3. Huấn Luyện

```bash
cd c:\DATN\phan_mem
python train.py --config configs/config.yaml
```

Kết quả lưu tại:

```text
checkpoints/best_model.pt
checkpoints/last_model.pt
logs/train_metrics.json
```

## A.4. Chạy Inference

```bash
cd c:\DATN\phan_mem
python inference.py --config configs/config.yaml --source 0
```

Hoặc với video:

```bash
python inference.py --config configs/config.yaml --source path\to\video.mp4
```

## A.5. Chạy Web Dashboard

```bash
cd c:\DATN\phan_mem
python web_server.py --config configs/config.yaml
```

## A.6. Chạy Ứng Dụng Quét Mặt

```bash
cd c:\DATN
python quetmat.py
```

## A.7. Test Telegram

```bash
cd c:\DATN\phan_mem
python test_alert_telegram.py
```

---

# PHỤ LỤC B. CẤU HÌNH `.env` ĐỀ XUẤT

```text
PUBLIC_BASE_URL=
ALERT_ENABLE_TELEGRAM=true
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_IDS=
ALERT_ACK_SECRET=
ALERT_COOLDOWN_SECONDS=45
ALERT_ENABLE_TELEGRAM_REMINDERS=true
ALERT_REMINDER_MAX_COUNT=3
ALERT_REMINDER_INTERVAL_SECONDS=60
AUTH_USERNAME=
AUTH_PASSWORD=
FACE_AUTH_DATABASE=../data/database.json
FACE_AUTH_TOLERANCE=0.5
```

Lưu ý: không công khai token Telegram, mật khẩu WiFi, mật khẩu đăng nhập hoặc secret xác nhận cảnh báo.

---

# PHỤ LỤC C. GỢI Ý NỘI DUNG THUYẾT TRÌNH

Đề tài của em là xây dựng hệ thống phát hiện té ngã và cảnh báo khẩn cấp cho người cao tuổi bằng camera, trí tuệ nhân tạo và IoT. Mục tiêu của hệ thống không chỉ là nhận diện té ngã trên video, mà là tạo ra một quy trình cảnh báo hoàn chỉnh: camera phát hiện sự cố, dashboard cập nhật trạng thái, Telegram gửi ảnh cảnh báo, ESP32 phát âm thanh và người giám sát có thể xác nhận an toàn.

Trong hệ thống, em sử dụng YOLOv8 Pose pretrained để phát hiện người và trích xuất keypoint tư thế. YOLOv8 Pose không phải là model phát hiện té ngã, mà chỉ là model nền để lấy thông tin cơ thể. Từ bounding box và keypoint, em xây dựng bộ đặc trưng gồm 37 chiều mô tả hình học, tư thế và chuyển động. Chuỗi đặc trưng này được đưa vào mô hình GRU tự huấn luyện để phân loại các trạng thái như đi/đứng, ngồi, nằm chủ động, đang ngã và đã ngã.

Điểm quan trọng của hệ thống là em không chỉ dựa vào kết quả mô hình trên từng frame. Em xây dựng thêm state machine để xác nhận té ngã bằng nhiều bằng chứng theo thời gian như tốc độ rơi, giảm chiều cao cơ thể, tư thế nằm sau ngã và trạng thái bất động. Nhờ đó, hệ thống giảm báo nhầm trong các tình huống sinh hoạt như ngồi xuống, cúi người, nằm nghỉ hoặc đi sát camera.

Khi phát hiện nguy hiểm, hệ thống tạo snapshot cảnh báo, cập nhật dashboard, gửi Telegram kèm ảnh và link xác nhận an toàn. Đồng thời ESP32 đọc trạng thái từ web server và phát cảnh báo âm thanh tại chỗ. Kết quả huấn luyện tốt nhất đạt F1 lớp FALLING là 0.825 và fall_safety_score là 0.761. Hạn chế hiện tại là dữ liệu test còn nhỏ và một số lớp chưa đủ mẫu, nên hướng phát triển là mở rộng dữ liệu thực tế, cải thiện phân biệt nằm chủ động với té ngã và tối ưu triển khai lâu dài.

---

# PHỤ LỤC D. CÂU HỎI BẢO VỆ CÓ THỂ GẶP

## Câu 1. YOLOv8 Pose có phải model phát hiện té ngã không?

Không. YOLOv8 Pose chỉ dùng để phát hiện người và keypoint tư thế. Phần phát hiện té ngã của đề tài là bộ đặc trưng 37 chiều, mô hình GRU tự huấn luyện và state machine xác nhận theo thời gian.

## Câu 2. Vì sao cần mô hình chuỗi thời gian?

Vì té ngã là một quá trình chuyển động. Nếu chỉ nhìn một frame, hệ thống dễ nhầm người nằm nghỉ với người đã ngã. Chuỗi thời gian giúp mô hình thấy được quá trình từ đứng/đi sang rơi xuống và nằm sau ngã.

## Câu 3. Vì sao cần state machine nếu đã có GRU?

GRU đưa ra xác suất trạng thái, nhưng trong thực tế vẫn có báo nhầm. State machine kiểm tra thêm tốc độ rơi, tư thế, thời gian và bằng chứng liên tiếp để quyết định ổn định hơn.

## Câu 4. Hệ thống xử lý báo nhầm như thế nào?

Hệ thống chặn các tình huống như ngồi, cúi, nằm chủ động, đi sát camera hoặc mất track ngắn bằng các điều kiện trong state machine và bộ lọc cảnh báo trong `AlertManager`.

## Câu 5. Đề tài có điểm gì khác so với chỉ chạy model có sẵn?

Đề tài không dùng YOLO để kết luận té ngã. YOLO chỉ trích xuất pose. Phần phân loại trạng thái là mô hình GRU tự huấn luyện, kết hợp state machine, web dashboard, Telegram, xác thực khuôn mặt và ESP32.

## Câu 6. Hạn chế lớn nhất hiện tại là gì?

Hạn chế lớn nhất là dữ liệu test độc lập còn nhỏ và một số lớp chưa đủ mẫu, đặc biệt là BENDING và UNCONSCIOUS. Vì vậy, hệ thống cần thêm dữ liệu thực tế để đánh giá và cải thiện.

## Câu 7. Nếu mất mạng thì hệ thống có hoạt động không?

Nếu camera và web server chạy local thì dashboard vẫn có thể hoạt động trong mạng nội bộ. Tuy nhiên, Telegram và TTS online trên ESP32 cần Internet. Hướng phát triển là thêm kênh cảnh báo dự phòng và âm thanh offline trên ESP32.

---

# PHỤ LỤC E. DANH MỤC HÌNH ẢNH NÊN CHÈN VÀO BÁO CÁO WORD

**Bảng E.1. Danh mục hình ảnh đề xuất chèn vào báo cáo Word**

| Hình | Nội dung đề xuất |
|---|---|
| Hình 1 | Sơ đồ kiến trúc tổng thể hệ thống |
| Hình 2 | Luồng xử lý từ camera đến cảnh báo |
| Hình 3 | YOLOv8 Pose phát hiện người và keypoint |
| Hình 4 | Giao diện đăng nhập |
| Hình 5 | Giao diện xác thực khuôn mặt |
| Hình 6 | Dashboard giám sát 4 camera |
| Hình 7 | Ảnh snapshot cảnh báo té ngã |
| Hình 8 | Tin nhắn Telegram cảnh báo |
| Hình 9 | ESP32 và module MAX98357A |
| Hình 10 | Biểu đồ kết quả huấn luyện |
| Hình 11 | Sơ đồ dùng DroidCam làm camera không dây |
| Hình 12 | Sơ đồ truy cập dashboard qua Cloudflare/tên miền |
