# app.py (File này sẽ được triển khai lên Render.com)

import requests
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, firestore
from sklearn.feature_extraction.text import TfidfVectorizer
from unidecode import unidecode
import os 
import json
from datetime import datetime, timedelta 

from sklearn.metrics.pairwise import cosine_similarity 

app = Flask(__name__)
# Cấu hình CORS: Cho phép ứng dụng React của bạn gọi API.
# Trong môi trường production, hãy thay thế "*" bằng domain của React app của bạn.
# Ví dụ: CORS(app, origins=["https://your-react-app-domain.com", "http://localhost:5173"])
CORS(app) 

# --- Cấu hình Firebase Admin SDK cho môi trường triển khai ---
# Render sẽ đọc Service Account Key từ biến môi trường GCP_SERVICE_ACCOUNT_KEY_JSON.
# Đảm bảo Service Account có quyền "Cloud Datastore Viewer" hoặc "Firestore Reader".
if not firebase_admin._apps:
    try:
        if os.getenv('GOOGLE_APPLICATION_CREDENTIALS'): 
            cred = credentials.ApplicationDefault()
            firebase_admin.initialize_app(cred)
            print("API: Firebase Admin SDK khởi tạo với Application Default Credentials.")
        elif os.getenv('GCP_SERVICE_ACCOUNT_KEY_JSON'): 
            cred_json = json.loads(os.getenv('GCP_SERVICE_ACCOUNT_KEY_JSON'))
            cred = credentials.Certificate(cred_json)
            firebase_admin.initialize_app(cred)
            print("API: Firebase Admin SDK khởi tạo từ biến môi trường JSON.")
        else:
            print("API: Không tìm thấy credentials trong biến môi trường. Vui lòng kiểm tra cấu hình.")
    except Exception as e:
        print(f"API: Lỗi khi khởi tạo Firebase Admin SDK: {e}. Vui lòng kiểm tra Service Account.")

db = firestore.client() if firebase_admin._apps else None

# --- Biến toàn cục để lưu DataFrame và các mô hình TF-IDF ---
ALL_MOVIES_DF = pd.DataFrame()
tfidf_vectorizer = None
movie_feature_matrix = None

# --- Hàm lấy lịch sử xem phim từ Firestore ---
def get_watch_history_from_firestore(user_id):
    history = []
    if not firebase_admin._apps: 
        print("API: Firebase Admin SDK chưa được khởi tạo. Không thể lấy lịch sử xem.")
        return history
    if db:
        try:
            history_ref = db.collection('users').document(user_id).collection('watchHistory')
            docs = history_ref.stream() 
            
            for doc_item in docs:
                data = doc_item.to_dict()
                genres_from_history = data.get('genres', [])
                movie_slug = data.get('slug')
                last_watched_episode_slug = data.get('lastWatchedEpisodeSlug')
                last_watched_episode_name = data.get('lastWatchedEpisodeName')

                history.append({
                    'movie_id': data.get('movieId'),
                    'title': data.get('title'),
                    'genres': genres_from_history,
                    'slug': movie_slug,
                    'poster_url': data.get('poster_url'),
                    'thumb_url': data.get('thumb_url'),
                    'year': data.get('year'),
                    'lastWatchedEpisodeSlug': last_watched_episode_slug,
                    'lastWatchedEpisodeName': last_watched_episode_name,
                    'total_watched_duration_seconds': data.get('total_watched_duration_seconds', 0),
                    'is_fully_watched': data.get('is_fully_watched', False)
                })
        except Exception as e:
            print(f"API: Lỗi khi lấy lịch sử xem phim từ Firestore: {e}")
    else:
        print("API: Firestore client chưa được khởi tạo. Kiểm tra lại Firebase Admin SDK.")
    return history

# --- Hàm: Đọc dữ liệu phim đã làm giàu từ Firestore và xây dựng mô hình ---
def load_movies_from_firestore_and_build_model(collection_name='enrichedMovies'):
    """
    Tải tất cả dữ liệu phim đã làm giàu từ Firestore và xây dựng TF-IDF model.
    """
    global ALL_MOVIES_DF, tfidf_vectorizer, movie_feature_matrix
    
    if not db:
        print("API: Firestore client chưa khả dụng để tải phim làm giàu.")
        ALL_MOVIES_DF = pd.DataFrame() 
        tfidf_vectorizer = None
        movie_feature_matrix = None
        return ALL_MOVIES_DF
    
    print("API: Đang tải toàn bộ dữ liệu phim đã làm giàu từ Firestore cho mô hình gợi ý...")
    movies_data = []
    try:
        docs = db.collection(collection_name).stream()
        for doc_item in docs:
            if doc_item.id != 'metadata': 
                movies_data.append(doc_item.to_dict())
        
        ALL_MOVIES_DF = pd.DataFrame(movies_data)
        if not ALL_MOVIES_DF.empty:
            tfidf_vectorizer = TfidfVectorizer(stop_words='english', min_df=5)
            movie_feature_matrix = tfidf_vectorizer.fit_transform(ALL_MOVIES_DF['combined_features'])
            print(f"API: Đã tải {len(ALL_MOVIES_DF)} phim từ Firestore. TF-IDF sẵn sàng.")
        else:
            print("API: Không có phim nào trong collection 'enrichedMovies'.")
            tfidf_vectorizer = None
            movie_feature_matrix = None
    except Exception as e:
        print(f"API: Lỗi khi tải phim làm giàu từ Firestore: {e}")
        ALL_MOVIES_DF = pd.DataFrame() 
        tfidf_vectorizer = None
        movie_feature_matrix = None
    
    return ALL_MOVIES_DF

# --- Hàm Gợi ý phim Content-Based Filtering ---
def recommend_movies_content_based(user_id, top_n=10, min_watch_duration_seconds=60, max_movies_for_profile=50): 
    """
    Gợi ý phim dựa trên nội dung đã xem của người dùng.
    Chỉ lấy các phim có thời gian xem nhiều nhất để tính toán hồ sơ sở thích.
    """
    if tfidf_vectorizer is None or movie_feature_matrix is None or ALL_MOVIES_DF.empty:
        print("API: Hệ thống gợi ý chưa sẵn sàng: Dữ liệu phim hoặc mô hình chưa được chuẩn bị.")
        return ALL_MOVIES_DF.head(top_n)[['slug', 'name', 'poster_url', 'thumb_url', 'year']].to_dict(orient='records')

    watch_history = get_watch_history_from_firestore(user_id)
    
    user_watched_movie_features = []
    watched_movie_slugs = set()

    meaningful_watched_movies_from_history = []
    for item in watch_history:
        if item.get('total_watched_duration_seconds', 0) >= min_watch_duration_seconds:
            meaningful_watched_movies_from_history.append(item)
            
    meaningful_watched_movies_from_history.sort(key=lambda x: x.get('total_watched_duration_seconds', 0), reverse=True)

    movies_for_profile_building = meaningful_watched_movies_from_history[:max_movies_for_profile]

    if not movies_for_profile_building:
        print(f"API: Người dùng {user_id} chưa có phim nào xem đủ thời lượng ({min_watch_duration_seconds}s) hoặc không có phim trong top {max_movies_for_profile} đã lọc. Gợi ý top phim mới nhất.")
        return ALL_MOVIES_DF.head(top_n)[['slug', 'name', 'poster_url', 'thumb_url', 'year']].to_dict(orient='records')


    for item in movies_for_profile_building: 
        watched_slug = item.get('slug')
        if watched_slug:
            watched_movie_slugs.add(watched_slug) 

            watched_movie_row = ALL_MOVIES_DF[ALL_MOVIES_DF['slug'] == watched_slug]
            if not watched_movie_row.empty:
                user_watched_movie_features.append(watched_movie_row['combined_features'].iloc[0])
            else:
                print(f"API: Phim đã xem (đủ thời lượng) '{watched_slug}' không tìm thấy trong dữ liệu phim tổng thể.")
        else:
             print(f"API: Không có slug cho phim trong lịch sử xem đủ thời lượng: {item}")

    if not user_watched_movie_features:
        print(f"API: Người dùng {user_id} có lịch sử nhưng không có phim nào hợp lệ để gợi ý CBF. Gợi ý top phim mới nhất.")
        return ALL_MOVIES_DF.head(top_n)[['slug', 'name', 'poster_url', 'thumb_url', 'year']].to_dict(orient='records')

    user_profile_vector = tfidf_vectorizer.transform([' '.join(user_watched_movie_features)])

    similarities = cosine_similarity(user_profile_vector, movie_feature_matrix).flatten()

    similarity_df = ALL_MOVIES_DF.copy()
    similarity_df['similarity'] = similarities

    unwatched_movies_df = similarity_df[~similarity_df['slug'].isin(watched_movie_slugs)]

    if unwatched_movies_df.empty:
        print(f"API: Người dùng {user_id} đã xem tất cả các phim khả dụng. Gợi ý top phim mới nhất.")
        return ALL_MOVIES_DF.head(top_n)[['slug', 'name', 'poster_url', 'thumb_url', 'year']].to_dict(orient='records')

    recommended_movies = unwatched_movies_df.sort_values(by='similarity', ascending=False).head(top_n)

    return recommended_movies[['slug', 'name', 'poster_url', 'thumb_url', 'year', 'genres_slugs', 'similarity']].to_dict(orient='records')


# --- Tải dữ liệu phim đã làm giàu một lần khi API khởi động ---
print("API: Khởi tạo tải dữ liệu phim khi API khởi động...")
load_movies_from_firestore_and_build_model(collection_name='enrichedMovies')


# --- API Endpoints ---
@app.route('/')
def home():
    return "API Gợi ý phim đang hoạt động!"

@app.route('/recommend', methods=['POST'])
def get_recommendations_api(): # <-- Định nghĩa đúng một lần duy nhất
    user_id = request.json.get('userId')
    if not user_id:
        return jsonify({"error": "Cần có userId để gợi ý phim. Vui lòng đăng nhập."}), 400

    print(f"API: Nhận yêu cầu gợi ý cho User ID: {user_id}")

    # Đảm bảo dữ liệu phim đã được load và mô hình sẵn sàng
    if ALL_MOVIES_DF.empty or tfidf_vectorizer is None or movie_feature_matrix is None:
        print("API: Dữ liệu phim hoặc mô hình chưa sẵn sàng. Đang thử tải lại từ Firestore...")
        loaded_df = load_movies_from_firestore_and_build_model()
        if loaded_df.empty:
            return jsonify({"error": "Hệ thống gợi ý chưa sẵn sàng. Vui lòng thử lại sau."}), 503
    
    recommendations = recommend_movies_content_based(user_id, top_n=10)
    return jsonify({"recommendations": recommendations})

# Dòng này không cần thiết khi triển khai lên Render.com (hoặc Cloud Run) vì Gunicorn sẽ quản lý cổng
# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))