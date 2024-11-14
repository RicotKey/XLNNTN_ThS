import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize, word_tokenize
import re
nltk.download('punkt_tab')

# Đọc nội dung từ file .txt
file_path = './text/d070f'  # Đổi 'path/to/your/file.txt' thành đường dẫn đến file của bạn
with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read()

### TIỀN XỬ LÝ ###
# 1. Tạo dictionary để lưu thông tin câu và các thẻ XML
sentences_dict = {}
sentence_list = text.split('\n')

for index, sentence in enumerate(sentence_list):
    if sentence.strip():
        # Tách các thẻ XML và nội dung câu
        xml_tags = re.findall(r"<.*?>", sentence)
        content = re.sub(r"<.*?>", "", sentence).strip()
        
        # Sử dụng chỉ số hoặc docid làm key
        docid_match = re.search(r'docid="([^"]+)"', sentence)
        key = docid_match.group(1) if docid_match else f"sent_{index}"
        
        # Lưu toàn bộ câu gốc và nội dung vào dictionary
        sentences_dict[key] = {
            'original_sentence': sentence.strip(),
            'content': content
        }

# Kiểm tra danh sách các câu đã lưu
print("Danh sách các câu (key-value):")
for key, value in sentences_dict.items():
    print(f"{key}: {value}")

### TÍNH TOÁN TF-IDF ###
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform([value['content'] for value in sentences_dict.values()])

# 3. Xếp hạng các câu dựa trên điểm TF-IDF trung bình
sentence_scores = X.sum(axis=1)
sentence_scores = np.array(sentence_scores).flatten()

# 4. Chọn 2 câu có điểm cao nhất để tóm tắt
top_sentence_indices = sentence_scores.argsort()[-2:][::-1]



### TÍNH TF-IDF ###
### XÂY DỰNG ĐỒ THỊ ###
### SỬ DỤNG PAGE RANK ###
### HẬU XỬ LÝ ###
# Tạo tóm tắt bằng cách lấy lại câu gốc từ dictionary
summary = []
keys = list(sentences_dict.keys())
for i in top_sentence_indices:
    key = keys[i]
    original_sentence = sentences_dict[key]['original_sentence']
    summary.append(original_sentence)

# Kết quả tóm tắt
print("\nTóm tắt văn bản:")
for sentence in summary:
    print(sentence)


