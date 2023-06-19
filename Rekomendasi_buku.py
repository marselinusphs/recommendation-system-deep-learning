import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
from keras.models import load_model


# Memuat model
model = tf.keras.models.load_model('model.keras', compile=False)

Dataset_buku = './data_sets'

# @st.cache_data

def prepare_data():
    
    Dataset_buku = './data_sets'
    with st.spinner('Memuat data...'):
        time.sleep(3)  # Durasi loading selama 3 detik
        buku = pd.read_csv(Dataset_buku+'/books.csv')
        genre_buku = pd.read_csv(Dataset_buku+'/book_tags.csv')
        ket_genre = pd.read_csv(Dataset_buku+'/tags.csv')
        user_read = pd.read_csv(Dataset_buku+'/to_read.csv')
        data_rating = pd.read_csv(Dataset_buku+'/ratings.csv')

    # Menggabungkan seluruh TagID pada kategori buku
    semua_tag = np.concatenate((
        genre_buku.tag_id.unique(),
        ket_genre.tag_id.unique()
    ))

    # Mengurutkan data dan menghapus data yang sama
    semua_tag = np.sort(np.unique(semua_tag))

    print('Jumlah seluruh data genre berdasarkan tag_id:', len(semua_tag))

    data_buku = pd.merge(data_rating, buku , on='book_id', how='left')
    data_buku.isnull().sum()

    # Menghitung jumlah rating berdasarkan book_id
    rating_per_book = data_buku.groupby('book_id').sum()

    all_rating = data_rating

    # Menggabungkan data rating dengan penulis, judul, dan tahun buku berdasarkan book_id
    all_book_rating = pd.merge(data_rating, data_buku[['book_id', 'authors', 'title', 'original_publication_year']], on='book_id', how='left')

    # Memeriksa missing value
    all_book_rating.isnull().sum()

    # Menghapus missing value dengan fungsi dropna()
    book_rating = all_book_rating.dropna()

    # Memeriksa kembali Missing Value
    book_rating.isnull().sum()

    # Menghapus data yang sama berdasarkan book id
    preparation = book_rating.drop_duplicates('book_id')

    # Mengonversi data series menjadi dalam bentuk list
    book_id = preparation['book_id'].tolist()
    book_title = preparation['title'].tolist()
    book_author = preparation['authors'].tolist()
    book_year = preparation['original_publication_year'].tolist()

    # Membuat dictionary untuk data 'book_id', 'book_title', 'book_author' dan 'book_year'
    book_data = pd.DataFrame({
        'id_buku': book_id,
        'judul_buku': book_title,
        'penulis': book_author,
        'tahun_rilis': book_year
    })

    tf = TfidfVectorizer()
    tf.fit(book_data['penulis'])
    tfidf_matrix = tf.fit_transform(book_data['penulis'])

    cosine_sim = cosine_similarity(tfidf_matrix)
    cosine_sim_df = pd.DataFrame(cosine_sim, index=book_data['judul_buku'], columns=book_data['judul_buku'])

    return book_data, cosine_sim_df

def book_recommendations(judul_buku, similarity_data, items, k=5):
    index = similarity_data.loc[:, judul_buku].to_numpy().argpartition(range(-1, -k, -1))
    closest = similarity_data.columns[index[-1:-(k+2):-1]]
    closest = closest.drop(judul_buku, errors='ignore')
    return pd.DataFrame(closest).merge(items).head(k)

def get_user_data(user_id, book_data):
    
    Dataset_buku = './data_sets'

    ratings_data = pd.read_csv(Dataset_buku + '/ratings.csv')

    df = ratings_data
    id_buku = df['book_id'].unique().tolist()

    book_to_book_encoded = {i: x for i, x in enumerate(id_buku)}

    user_ratings = ratings_data[ratings_data['user_id'] == user_id]
    book_ids_read_by_user = user_ratings['book_id'].values

    book_not_read = book_data[~book_data['id_buku'].isin(book_ids_read_by_user)]['id_buku']
    book_not_read = list(
        set(book_not_read)
        .intersection(set(book_to_book_encoded.keys()))
    )

    book_not_read = [[book_to_book_encoded.get(x)] for x in book_not_read]

    return user_ratings, book_not_read

def show_user_recommendations(user_ratings, book_not_read, book_data):
    
    Dataset_buku = './data_sets'

    ratings_data = pd.read_csv(Dataset_buku + '/ratings.csv')

    df = ratings_data
    id_reader = df['user_id'].unique().tolist()
    id_buku = df['book_id'].unique().tolist()
    
    user_to_user_encoded = {x: i for i, x in enumerate(id_reader)}
    book_encoded_to_book = {i: x for i, x in enumerate(id_buku)}

    user_encoder = user_to_user_encoded.get(user_ratings['user_id'].values[0])
    user_book_array = np.hstack(
        ([[user_encoder]] * len(book_not_read), book_not_read)
    )

    rating_buku = model.predict(user_book_array).flatten()

    top_ratings_indices = rating_buku.argsort()[-10:][::-1]
    recommended_book_ids = [
        book_encoded_to_book.get(book_not_read[x][0]) for x in top_ratings_indices
    ]

    st.write('Menampilkan Rekomendasi Buku untuk Pembaca dengan User ID:', user_ratings['user_id'].values[0])
    st.write('===' * 15)
    st.write('Daftar Rekomendasi Buku dengan rating tinggi dari pembaca')
    st.write('----' * 15)

    top_book_user = (
        user_ratings.sort_values(
            by='rating',
            ascending=False
        )
        .head(5)
        .book_id.values
    )

    book_df_rows = book_data[book_data['id_buku'].isin(top_book_user)]
    for row in book_df_rows.itertuples():
        st.write(row.penulis, ':', row.judul_buku)

    st.write('----' * 15)
    st.write('Daftar Top 10 Buku yang Direkomendasikan')
    st.write('----' * 15)

    recommended_book = book_data[book_data['id_buku'].isin(recommended_book_ids)]
    for row in recommended_book.itertuples():
        st.write(row.penulis, ':', row.judul_buku)


def main():
    st.title('Sistem Rekomendasi Buku')

    book_data, cosine_sim_df = prepare_data()

    judul_buku_input = st.text_input('Masukkan judul buku:')
    rekomendasi_jumlah = st.slider('Jumlah rekomendasi:', 1, 10, 5)

    if st.button('Rekomendasikan'):
        rekomendasi = book_recommendations(judul_buku_input, cosine_sim_df, book_data, k=rekomendasi_jumlah)
        st.write('Rekomendasi Buku:')
        st.dataframe(rekomendasi)


    user_id_input = st.text_input('Masukkan user ID:')
    
    if st.button('Tampilkan Rekomendasi User'):
        # df = pd.read_csv(Dataset_buku + '/ratings.csv')
        user_ratings, book_not_read = get_user_data(int(user_id_input), book_data)
        show_user_recommendations(user_ratings, book_not_read, book_data)


# Memanggil fungsi show_data saat aplikasi Streamlit dijalankan
if __name__ == '__main__':
    main()
