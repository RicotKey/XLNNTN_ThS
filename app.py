import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import re
import os
nltk.download('punkt_tab')
nltk.download('stopwords')

# Đọc nội dung từ file gốc
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def preprocess_text(text):
    sentences_dict = {}
    sentence_list = text.split('\n')

    # Sử dụng set để loại bỏ các câu trùng lặp
    unique_sentences = set()

    for index, sentence in enumerate(sentence_list):
        if sentence.strip():  # Chỉ xử lý câu không trống
            # Loại bỏ tất cả các thẻ XML/HTML như <s>, <TEXT>, docid, num, wdcount, v.v.
            sentence_cleaned = re.sub(r"<.*?>", "", sentence).strip()

            # Bỏ qua câu nếu nó chứa dấu ngoặc kép (câu thoại)
            if "``" in sentence_cleaned or "''" in sentence_cleaned:
                continue  # Loại bỏ dấu `` và ''

            # Kiểm tra nếu câu sau khi làm sạch không trống
            if not sentence_cleaned:
                continue  # Bỏ qua câu không có nội dung

            # Chuyển tất cả các ký tự thành chữ thường để chuẩn hóa
            sentence_cleaned = sentence_cleaned.lower()

            # Tạo key duy nhất cho mỗi câu sử dụng chỉ số câu
            key = f"sent_{index}"

            # Kiểm tra xem câu này đã tồn tại trong set chưa
            if sentence_cleaned not in unique_sentences:
                unique_sentences.add(sentence_cleaned)

                # Lưu toàn bộ câu gốc và nội dung vào dictionary
                sentences_dict[key] = {
                    'original_sentence': sentence.strip(),
                    'content': sentence_cleaned
                }

    return sentences_dict

# Tính toán TF-IDF cho các câu trong văn bản
def compute_tfidf(sentences_dict):
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.85, min_df=2, ngram_range=(1, 3))
    X = vectorizer.fit_transform([value['content'] for value in sentences_dict.values()])
    return X

# Xây dựng ma trận tương đồng
def build_adjacency_matrix(cosine_similarities, threshold=0.4):
    adj_matrix = np.zeros(cosine_similarities.shape)
    for i in range(len(cosine_similarities)):
        for j in range(i + 1, len(cosine_similarities)):
            if cosine_similarities[i, j] > threshold:
                adj_matrix[i, j] = cosine_similarities[i, j]
                adj_matrix[j, i] = cosine_similarities[i, j]
    return adj_matrix

# Tính toán PageRank
def compute_pagerank(adj_matrix, damping_factor=0.85, max_iter=100, tolerance=1e-6):
    pagerank_scores = np.ones(adj_matrix.shape[0]) / adj_matrix.shape[0]
    for _ in range(max_iter):
        prev_pagerank_scores = pagerank_scores.copy()
        pagerank_scores = (1 - damping_factor) / len(pagerank_scores) + damping_factor * np.dot(adj_matrix, pagerank_scores)
        
        if np.linalg.norm(pagerank_scores - prev_pagerank_scores, 1) < tolerance:
            break
    return pagerank_scores

# Chọn các câu có PageRank cao nhất
def select_top_sentences(sentences_dict, pagerank_scores, top_percentage=20):
    # Tính số câu tối đa dựa trên phần trăm cho trước
    total_sentences = len(sentences_dict)
    top_n = max(1, int(total_sentences * top_percentage / 100))  # Đảm bảo lấy ít nhất 1 câu

    # Lấy chỉ số của các câu có điểm PageRank cao nhất
    top_sentence_indices = np.argsort(pagerank_scores)[-top_n:][::-1]
    keys = list(sentences_dict.keys())

    selected_sentences = []
    for i in top_sentence_indices:
        key = keys[i]
        sentence = sentences_dict[key]
        sentence['index'] = i  # Thêm index của câu vào để duy trì thứ tự
        selected_sentences.append(sentence)

    # Sắp xếp lại theo thứ tự ban đầu
    selected_sentences_sorted = sorted(selected_sentences, key=lambda x: x['index'])

    # Lấy câu gốc để tạo tóm tắt
    summary = [sentence['original_sentence'] for sentence in selected_sentences_sorted]
    return summary

# Tính tỷ lệ chính xác của tóm tắt so với tóm tắt từ file
def calculate_accuracy(summary, summary_sentences):
    summary_set = set(summary)
    summary_sentences_set = set(summary_sentences)
    correct_count = len(summary_set.intersection(summary_sentences_set))
    accuracy = correct_count / len(summary_sentences_set) * 100 if len(summary_sentences_set) > 0 else 0
    return accuracy

# Hàm chính thực hiện toàn bộ quy trình tóm tắt văn bản
def main():
    # Nhận mã tài liệu từ người dùng
    document_code = input("Nhập mã tài liệu: ")

    # Đường dẫn đến file văn bản gốc và tóm tắt
    file_path = f'./text/{document_code}'  # Sử dụng mã tài liệu để xác định file
    summary_file_path = f'./sum/{document_code}'  # Sử dụng mã tài liệu để xác định file

    # Kiểm tra xem file có tồn tại không trước khi thực hiện đọc
    if not os.path.exists(file_path) or not os.path.exists(summary_file_path):
        print("Lỗi: Không tìm thấy file văn bản hoặc tóm tắt tại đường dẫn chỉ định.")
        return

    # Đọc file văn bản gốc và tóm tắt
    text = read_file(file_path)
    summary_text = read_file(summary_file_path)

    # Tiền xử lý văn bản và loại bỏ trùng lặp
    sentences_dict = preprocess_text(text)

    # Tính toán TF-IDF
    X = compute_tfidf(sentences_dict)

    # Tính toán độ tương đồng cosine giữa các câu
    cosine_similarities = cosine_similarity(X)

    # Xây dựng ma trận tương tác
    adj_matrix = build_adjacency_matrix(cosine_similarities)

    # Tính toán PageRank
    pagerank_scores = compute_pagerank(adj_matrix)

    # Chọn các câu có PageRank cao nhất
    summary = select_top_sentences(sentences_dict, pagerank_scores)

    # In tóm tắt
    print("\nTóm tắt văn bản:")
    for sentence in summary:
        print(sentence)

    # Tách câu trong tóm tắt từ file 'sum'
    summary_sentences = summary_text.split('\n')
    summary_sentences = [sent.strip() for sent in summary_sentences if sent.strip()]

    # Tính tỷ lệ chính xác của tóm tắt
    accuracy = calculate_accuracy(summary, summary_sentences)
    print(f"Tỷ lệ chính xác của tóm tắt (không theo thứ tự): {accuracy:.2f}%")

# Chạy hàm chính
if __name__ == "__main__":
    main()
