import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.express as px

from utils import (
    fetch_poster, 
    recommend, 
    improved_recommendations
)

st.set_page_config(page_title="Movie Recommender & Analytics", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://getwallpapers.com/wallpaper/full/4/4/c/327072.jpg");
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_data
def load_data():
    with open('data/movies_df.pkl', 'rb') as f:
        movies = pickle.load(f)
    with open('data/movies_df_rec.pkl', 'rb') as f:
        movies_rec = pickle.load(f)
    with open('data/cosine_sim.pkl', 'rb') as f:
        similarity = pickle.load(f)
    return movies, movies_rec, similarity

movies, movies_rec, similarity = load_data()

required_columns = [
    'budget','revenue','vote_average','runtime','popularity',
    'genres','cast','director','id','title', 'overview'
]
for col in required_columns:
    if col not in movies.columns:
        st.error(f"Столбец '{col}' отсутствует в DataFrame 'movies'.")
    if col not in movies_rec.columns:
        st.error(f"Столбец '{col}' отсутствует в DataFrame 'movies_rec'.")

numeric_cols = ['budget','revenue','vote_average','runtime','popularity']
for col in numeric_cols:
    if col in movies.columns:
        movies[col] = pd.to_numeric(movies[col], errors='coerce')
    if col in movies_rec.columns:
        movies_rec[col] = pd.to_numeric(movies_rec[col], errors='coerce')

if 'release_date' in movies.columns:
    movies['release_date'] = pd.to_datetime(movies['release_date'], errors='coerce')
    movies['year'] = movies['release_date'].dt.year
else:
    movies['year'] = pd.to_numeric(movies.get('year', pd.Series([np.nan]*len(movies))), errors='coerce')

if 'release_date' in movies_rec.columns:
    movies_rec['release_date'] = pd.to_datetime(movies_rec['release_date'], errors='coerce')
    movies_rec['year'] = movies_rec['release_date'].dt.year
else:
    movies_rec['year'] = pd.to_numeric(movies_rec.get('year', pd.Series([np.nan]*len(movies_rec))), errors='coerce')

movies = movies[movies['year'] <= 2017].reset_index(drop=True)
movies_rec = movies_rec[movies_rec['year'] <= 2017].reset_index(drop=True)

def extract_list(field):
    
    if isinstance(field, list):
        return [str(x).strip() for x in field if isinstance(x, str)]
    elif isinstance(field, str):
        return [x.strip() for x in field.split(',')]
    return []

movies['genres'] = movies['genres'].apply(extract_list) if 'genres' in movies.columns else [[]]
movies['cast'] = movies['cast'].apply(extract_list) if 'cast' in movies.columns else [[]]
movies['director'] = movies['director'].apply(extract_list) if 'director' in movies.columns else [[]]

movies_rec['genres'] = movies_rec['genres'].apply(extract_list) if 'genres' in movies_rec.columns else [[]]
movies_rec['cast'] = movies_rec['cast'].apply(extract_list) if 'cast' in movies_rec.columns else [[]]
movies_rec['director'] = movies_rec['director'].apply(extract_list) if 'director' in movies_rec.columns else [[]]


tab1, tab2 = st.tabs(["Рекомендательная система", "Общая аналитика фильмов"])

# ----------------------------------------------------------------------------------
#                ТАБ 1: РЕКОМЕНДАТЕЛЬНАЯ СИСТЕМА
# ----------------------------------------------------------------------------------
with tab1:
    st.markdown("<h2 style='text-align: center;'>Рекомендательная система фильмов</h2>", unsafe_allow_html=True)

    
    if 'title' in movies_rec.columns:
        movie_list = movies_rec['title'].dropna().values
    else:
        movie_list = []

    selected_movie = st.selectbox("Выберите или начните вводить название фильма:", movie_list, key="recommender_selectbox")

 
    cols_buttons = st.columns(2)
    with cols_buttons[0]:
        if st.button('Показать рекомендации', key='show_recommendations'):
            # Получаем обычные рекомендации
            rec_names, rec_posters = recommend(selected_movie, movies_rec, similarity)
            # Сохраняем рекомендации в сессии
            st.session_state['recommended_movies'] = rec_names
            st.session_state['recommended_posters'] = rec_posters

    with cols_buttons[1]:
        if st.button('Показать продвинутые рекомендации', key='show_advanced_recommendations'):
            # Получаем продвинутые рекомендации
            adv_recs = improved_recommendations(selected_movie, movies_rec, similarity)
            if adv_recs is not None and not adv_recs.empty:
                adv_names = adv_recs['title'].tolist()
                adv_posters = [fetch_poster(mid) for mid in adv_recs['id'].tolist()]
                # Сохраняем рекомендации в сессии
                st.session_state['recommended_movies'] = adv_names
                st.session_state['recommended_posters'] = adv_posters
            else:
                st.warning("Нет продвинутых рекомендаций.")

    # Отображение рекомендаций, если они есть
    if 'recommended_movies' in st.session_state and 'recommended_posters' in st.session_state:
        st.markdown("---")
        st.markdown("### Рекомендованные фильмы")

        rec_movies = st.session_state['recommended_movies']
        rec_posters = st.session_state['recommended_posters']
        num_recommendations = len(rec_movies)
        num_cols = 5
        cols = st.columns(num_cols)

        for i in range(num_recommendations):
            col_index = i % num_cols
            with cols[col_index]:
                poster = rec_posters[i]
                
                if poster:
                    st.image(poster, use_container_width=True)
                else:
                    st.image("https://via.placeholder.com/300x450?text=Нет+постера", use_container_width=True)
               
                st.text(rec_movies[i])

                
                mid = movies_rec[movies_rec['title'] == rec_movies[i]]['id'].values
                if len(mid) > 0:
                    mid = mid[0]
                else:
                    mid = None

              
                if mid:
                   
                    button_key = f"show_analytics_{mid}"
                    if st.button("Показать аналитику", key=button_key):
                        st.session_state['selected_movie_id'] = mid

        
        if 'selected_movie_id' in st.session_state and st.session_state['selected_movie_id']:
            selected_mid = st.session_state['selected_movie_id']
            selected_movie_data = movies_rec[movies_rec['id'] == selected_mid]
            if not selected_movie_data.empty:
                selected_movie_data = selected_movie_data.iloc[0]
                st.markdown("---")
                st.markdown("### Аналитика выбранного фильма")

              
                metric_cols = st.columns(4)
                with metric_cols[0]:
                    budget = selected_movie_data['budget']
                    if not pd.isna(budget) and budget >= 25000:
                        st.metric("Бюджет (USD)", f"{int(budget):,}")
                        budget_display = f"{int(budget):,}"
                    else:
                        st.metric("Бюджет (USD)", "нет данных")
                        budget_display = "нет данных"
                with metric_cols[1]:
                    revenue = selected_movie_data['revenue']
                    if not pd.isna(revenue) and revenue >= 25000:
                        st.metric("Сборы (USD)", f"{int(revenue):,}")
                        revenue_display = f"{int(revenue):,}"
                    else:
                        st.metric("Сборы (USD)", "нет данных")
                        revenue_display = "нет данных"
                with metric_cols[2]:
                    rating = selected_movie_data['vote_average']
                    if not pd.isna(rating):
                        st.metric("Рейтинг", f"{rating:.2f}")
                    else:
                        st.metric("Рейтинг", "нет данных")
                with metric_cols[3]:
                    runtime = selected_movie_data['runtime']
                    if not pd.isna(runtime) and runtime >= 30:
                        st.metric("Длительность (мин)", f"{int(runtime)}")
                    else:
                        st.metric("Длительность (мин)", "нет данных")

        
                st.markdown("#### Дополнительная информация")
                info_cols = st.columns(2)
                with info_cols[0]:
                    genres = ", ".join(selected_movie_data['genres']) if selected_movie_data['genres'] else "нет данных"
                    st.write(f"**Жанры:** {genres}")
                    cast = ", ".join(selected_movie_data['cast']) if selected_movie_data['cast'] else "нет данных"
                    st.write(f"**Актёры:** {cast}")
                with info_cols[1]:
                    director = ", ".join(selected_movie_data['director']) if selected_movie_data['director'] else "нет данных"
                    st.write(f"**Режиссёр:** {director}")
                    popularity = selected_movie_data['popularity']
                    if not pd.isna(popularity):
                        st.write(f"**Популярность:** {popularity:.2f}")
                    else:
                        st.write("**Популярность:** нет данных")

                # Описание (overview) без перевода
                overview = selected_movie_data['overview']
                if pd.notna(overview) and overview.strip() != "":
                    st.markdown("#### Описание:")
                    st.write(overview)
                else:
                    st.markdown("#### Описание:")
                    st.write("Описание отсутствует.")

                # ROI
                if (not pd.isna(budget) and budget >= 25000) and (not pd.isna(revenue) and revenue >= 25000):
                    roi = (revenue - budget) / budget
                    st.markdown(f"**ROI:** {roi:.2f}")
                else:
                    st.markdown("**ROI:** нет данных")

                # Место в рейтинге (год выпуска)
                release_year = selected_movie_data['year']
                if not pd.isna(release_year):
                    # Фильтрация фильмов того же года
                    same_year_movies = movies_rec[movies_rec['year'] == release_year].copy()

                    # Ранг по бюджету
                    if (not pd.isna(budget) and budget >= 25000):
                        same_year_budget = same_year_movies.dropna(subset=['budget']).sort_values(by='budget', ascending=False).reset_index(drop=True)
                        rank_budget = same_year_budget[same_year_budget['id'] == selected_movie_data['id']].index.tolist()
                        rank_budget = rank_budget[0] + 1 if rank_budget else "-"
                    else:
                        rank_budget = "-"

                    # Ранг по сборам
                    if (not pd.isna(revenue) and revenue >= 25000):
                        same_year_revenue = same_year_movies.dropna(subset=['revenue']).sort_values(by='revenue', ascending=False).reset_index(drop=True)
                        rank_revenue = same_year_revenue[same_year_revenue['id'] == selected_movie_data['id']].index.tolist()
                        rank_revenue = rank_revenue[0] + 1 if rank_revenue else "-"
                    else:
                        rank_revenue = "-"

                    # Ранг по рейтингу
                    if not pd.isna(rating):
                        same_year_rating = same_year_movies.dropna(subset=['vote_average']).sort_values(by='vote_average', ascending=False).reset_index(drop=True)
                        rank_rating = same_year_rating[same_year_rating['id'] == selected_movie_data['id']].index.tolist()
                        rank_rating = rank_rating[0] + 1 if rank_rating else "-"
                    else:
                        rank_rating = "-"

                    st.markdown("### Место в рейтинге (год выпуска)")
                    rank_cols = st.columns(3)
                    with rank_cols[0]:
                        st.info(f"Ранг по бюджету: {rank_budget}" if rank_budget != "-" else "Ранг по бюджету: -")
                    with rank_cols[1]:
                        st.info(f"Ранг по сборам: {rank_revenue}" if rank_revenue != "-" else "Ранг по сборам: -")
                    with rank_cols[2]:
                        st.info(f"Ранг по рейтингу: {rank_rating}" if rank_rating != "-" else "Ранг по рейтингу: -")
                else:
                    st.markdown("### Место в рейтинге (год выпуска)")
                    st.write("Данных о годе выпуска нет.")

# ----------------------------------------------------------------------------------
#                ТАБ 2: АНАЛИТИКА
# ----------------------------------------------------------------------------------
with tab2:
    st.markdown("<h2 style='text-align: center;'>Общаяаналитика (Dashboard)</h2>", unsafe_allow_html=True)
    st.markdown("Дашборд по имеющимся фильмам.")

    st.subheader("Ключевая статистика")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Всего фильмов", f"{len(movies):,}")
    with col2:
        avg_rating = movies['vote_average'].mean() if 'vote_average' in movies.columns else 0
        st.metric("Средний рейтинг", f"{avg_rating:.2f}")
    with col3:
        total_budget = movies['budget'].sum() if 'budget' in movies.columns else 0
        st.metric("Общий бюджет (USD)", f"{int(total_budget):,}")
    with col4:
        if 'revenue' in movies.columns and (movies['revenue'] > 0).any():
            total_revenue = movies['revenue'].sum()
            st.metric("Общие сборы (USD)", f"{int(total_revenue):,}")
        else:
            st.metric("Общие сборы (USD)", "нет данных")

    st.write("---")

    st.subheader("Фильтры")

    if 'genres' in movies.columns:
        df_gen_all = movies.explode('genres').dropna(subset=['genres'])
        genre_counts = df_gen_all['genres'].value_counts().reset_index()
        genre_counts.columns = ['genre', 'count']
        top_n_genres = 10
        genre_counts_top = genre_counts.head(top_n_genres)

        selected_genres = st.multiselect(
            "Выберите жанры для анализа",
            options=genre_counts['genre'].unique(),
            default=genre_counts_top['genre'].tolist(),
            key="analytics_multiselect_genres_tab2"
        )
    else:
        selected_genres = []
        st.warning("Столбец 'genres' не найден в датасете!")

    st.write("---")  

    fcol1, fcol2, fcol3, fcol4 = st.columns(4)
    with fcol1:
        if 'year' in movies.columns:
            min_year, max_year = int(movies['year'].min()), int(movies['year'].max())
            selected_year_range = st.slider("Года выпуска",
                                            min_value=min_year,
                                            max_value=max_year,
                                            value=(min_year, 2017),
                                            key="analytics_slider_year_tab2")
        else:
            selected_year_range = (1900, 2050)

    with fcol2:
        if 'vote_average' in movies.columns:
            min_rating, max_rating = float(movies['vote_average'].min()), float(movies['vote_average'].max())
            selected_rating_range = st.slider("Диапазон рейтингов",
                                              min_value=min_rating,
                                              max_value=max_rating,
                                              value=(min_rating, max_rating),
                                              key="analytics_slider_rating_tab2")
        else:
            selected_rating_range = (0, 10)

    with fcol3:
        if 'budget' in movies.columns:
            min_budget, max_budget = int(movies['budget'].min()), int(movies['budget'].max())
            selected_budget_range = st.slider("Диапазон бюджета (USD)",
                                              min_value=min_budget,
                                              max_value=max_budget,
                                              value=(min_budget, max_budget),
                                              key="analytics_slider_budget_tab2")
        else:
            selected_budget_range = (0, 1_000_000_000)

    with fcol4:
        if 'revenue' in movies.columns:
            min_revenue, max_revenue = int(movies['revenue'].min()), int(movies['revenue'].max())
            selected_revenue_range = st.slider("Диапазон сборов (USD)",
                                               min_value=min_revenue,
                                               max_value=max_revenue,
                                               value=(min_revenue, max_revenue),
                                               key="analytics_slider_revenue_tab2")
        else:
            selected_revenue_range = (0, 1_000_000_000)

    df_filtered = movies[
        (movies['year'] >= selected_year_range[0]) & (movies['year'] <= selected_year_range[1]) &
        (movies['vote_average'] >= selected_rating_range[0]) & (movies['vote_average'] <= selected_rating_range[1]) &
        (movies['budget'] >= selected_budget_range[0]) & (movies['budget'] <= selected_budget_range[1]) &
        (movies['revenue'] >= selected_revenue_range[0]) & (movies['revenue'] <= selected_revenue_range[1])
    ]

    if selected_genres:
        df_filtered = df_filtered[df_filtered['genres'].apply(lambda x: any(genre in selected_genres for genre in x))]

    st.write(f"**Фильмов после фильтрации:** {len(df_filtered):,}")
    st.write("---")

    if 'vote_average' in df_filtered.columns:
        st.subheader("Распределение рейтингов")

        df_vote = df_filtered['vote_average'].dropna()
        if not df_vote.empty:
            fig_hist = px.histogram(
                df_vote, 
                nbins=20,
                title="Гистограмма рейтингов",
                labels={'value': 'Рейтинг'},
                template="plotly_white"
            )
            st.plotly_chart(fig_hist, use_container_width=True, key='vote_average_hist_tab2')
        else:
            st.write("Нет данных для построения гистограммы рейтингов.")
    else:
        st.write("Нет данных для анализа рейтингов.")


    if 'budget' in df_filtered.columns and 'revenue' in df_filtered.columns:
        st.subheader("Бюджет и Сборы (Scatter Plot)")

        df_br = df_filtered[(df_filtered['budget'] >= 50000) & (df_filtered['revenue'] >= 50000)].copy()
        if not df_br.empty:
          
            df_br['ROI'] = (df_br['revenue'] - df_br['budget']) / df_br['budget']

            df_br = df_br[(df_br['ROI'] >= -1) & (df_br['ROI'] <= 10)]
            
            fig_scatter = px.scatter(
                df_br,
                x="budget",
                y="revenue",
                size="popularity" if "popularity" in df_br.columns else None,
                hover_data=["title"] if "title" in df_br.columns else [],
                labels={'budget': 'Бюджет (USD)', 'revenue': 'Сборы (USD)'},
                title="Бюджет vs Сборы (размер круга = популярность)",
                log_x=True, log_y=True,
                template="plotly_white"
            )
            st.plotly_chart(fig_scatter, use_container_width=True, key='budget_vs_revenue_scatter_tab2')
        else:
            st.write("Нет достаточных данных по бюджету и/или сборам (>= 50,000 USD).")

   
    if 'budget' in df_filtered.columns and 'revenue' in df_filtered.columns:
        st.subheader("Окупаемость (ROI)")

        df_roi = df_filtered[
            (df_filtered['budget'] >= 50000) & 
            (df_filtered['revenue'] >= 50000)
        ].copy()
        if not df_roi.empty:
           
            df_roi['ROI'] = (df_roi['revenue'] - df_roi['budget']) / df_roi['budget']
          
            df_roi = df_roi[(df_roi['ROI'] >= -1) & (df_roi['ROI'] <= 10)]
            
    
            if not df_roi['ROI'].dropna().empty:
                fig_roi_hist = px.histogram(
                    df_roi['ROI'], 
                    nbins=50,
                    title="Гистограмма ROI",
                    labels={'value': 'ROI (Сборы - Бюджет) / Бюджет'},
                    template="plotly_white"
                )
                st.plotly_chart(fig_roi_hist, use_container_width=True, key='roi_hist_tab2')
            else:
                st.write("Нет данных для построения гистограммы ROI.")
        else:
            st.write("Недостаточно данных для расчёта ROI (бюджет и сборы >= 50,000 USD).")

   
    st.subheader("Средний рейтинг от сборов")

    df_avg_rating_rev = df_filtered[
        (df_filtered['budget'] >= 50000) & 
        (df_filtered['revenue'] >= 50000)
    ].copy()
    if not df_avg_rating_rev.empty:
        try:
           
            df_avg_rating_rev['revenue_category'] = pd.qcut(df_avg_rating_rev['revenue'], q=10, duplicates='drop')
        except ValueError:
           
            df_avg_rating_rev['revenue_category'] = pd.qcut(df_avg_rating_rev['revenue'], q=5, duplicates='drop')
        
        
        df_avg_rating_rev['revenue_category'] = df_avg_rating_rev['revenue_category'].apply(
            lambda x: f"{int(x.left/1000)}k - {int(x.right/1000)}k"
        )

        df_rating_rev = df_avg_rating_rev.groupby('revenue_category')['vote_average'].mean().reset_index()

        fig_avg_rating_rev = px.scatter(
            df_rating_rev,
            x='revenue_category',
            y='vote_average',
            size='vote_average',
            labels={'revenue_category': 'Категория сборов (USD)', 'vote_average': 'Средний рейтинг'},
            title="Средний рейтинг по категориям сборов",
            template="plotly_white"
        )
        st.plotly_chart(fig_avg_rating_rev, use_container_width=True, key='avg_rating_rev_scatter_tab2')
    else:
        st.write("Недостаточно данных для анализа среднего рейтинга по сборам.")

  
    if 'year' in df_filtered.columns and 'vote_average' in df_filtered.columns:
        st.subheader("Средний рейтинг по годам")
        df_yearly = df_filtered.groupby('year')['vote_average'].mean().reset_index()
        if len(df_yearly) > 1:
            fig_line = px.line(
                df_yearly, 
                x='year', 
                y='vote_average',
                title="Изменение среднего рейтинга по годам",
                labels={'year': 'Год', 'vote_average': 'Средний рейтинг'},
                template="plotly_white"
            )
            st.plotly_chart(fig_line, use_container_width=True, key='avg_rating_line_tab2')
        else:
            st.write("Недостаточно данных для отображения тренда.")

   
    st.subheader("Анализ по жанрам")

    if 'genres' in movies.columns:
   
        df_genres = df_filtered.explode('genres').dropna(subset=['genres'])

        # Топ-10 жанров по количеству фильмов
        genre_counts = df_genres['genres'].value_counts().reset_index()
        genre_counts.columns = ['genre', 'count']
        top_n = 10
        genre_counts_top = genre_counts.head(top_n)


        if selected_genres:
            df_genres_selected = df_genres[df_genres['genres'].isin(selected_genres)]

            # Топ-10 жанров по количеству фильмов
            genre_counts_selected = df_genres_selected['genres'].value_counts().reset_index().head(top_n)
            genre_counts_selected.columns = ['genre', 'count']

            if not genre_counts_selected.empty:
                fig_genres = px.bar(
                    genre_counts_selected,
                    x='genre', 
                    y='count',
                    labels={'genre': 'Жанр', 'count': 'Количество фильмов'},
                    title=f"Топ {top_n} жанров (по количеству фильмов)",
                    template="plotly_white"
                )
                st.plotly_chart(fig_genres, use_container_width=True, key='genre_count_bar_tab2')
            else:
                st.write("Нет данных для выбранных жанров.")

            # Средний рейтинг по жанрам
            df_genre_rating = df_genres_selected.groupby('genres')['vote_average'].mean().reset_index()
            if not df_genre_rating.empty:
                fig_genre_rating = px.bar(
                    df_genre_rating, 
                    x='genres', 
                    y='vote_average',
                    labels={'genres': 'Жанр', 'vote_average': 'Средний рейтинг'},
                    title="Средний рейтинг по жанрам",
                    template="plotly_white"
                )
                st.plotly_chart(fig_genre_rating, use_container_width=True, key='genre_avg_rating_bar_tab2')
            else:
                st.write("Нет данных для расчёта среднего рейтинга по жанрам.")

            # Средние сборы по жанрам
            if 'revenue' in df_genres_selected.columns:
                df_genre_revenue = df_genres_selected.groupby('genres')['revenue'].mean().reset_index()
                if not df_genre_revenue.empty:
                    fig_genre_revenue = px.bar(
                        df_genre_revenue, 
                        x='genres', 
                        y='revenue',
                        labels={'genres': 'Жанр', 'revenue': 'Средние сборы (USD)'},
                        title="Средние сборы по жанрам",
                        template="plotly_white"
                    )
                    st.plotly_chart(fig_genre_revenue, use_container_width=True, key='genre_avg_revenue_bar_tab2')
                else:
                    st.write("Нет данных для расчёта средних сборов по жанрам.")
            else:
                st.write("Столбец 'revenue' не найден в датасете.")

            # Средний ROI по жанрам
     
            if 'budget' in df_genres_selected.columns and 'revenue' in df_genres_selected.columns:
                df_genres_selected_roi = df_genres_selected[
                    (df_genres_selected['budget'] >= 50000) & 
                    (df_genres_selected['revenue'] >= 50000)
                ].copy()
                if not df_genres_selected_roi.empty:
                    df_genres_selected_roi['ROI'] = (df_genres_selected_roi['revenue'] - df_genres_selected_roi['budget']) / df_genres_selected_roi['budget']
                    df_genres_selected_roi = df_genres_selected_roi[
                        (df_genres_selected_roi['ROI'] >= -1) & 
                        (df_genres_selected_roi['ROI'] <= 10)
                    ]

                    df_genre_roi = df_genres_selected_roi.groupby('genres')['ROI'].mean().reset_index()
                    if not df_genre_roi.empty:
                        fig_genres_roi = px.bar(
                            df_genre_roi, 
                            x='genres', 
                            y='ROI',
                            labels={'genres': 'Жанр', 'ROI': 'Средний ROI'},
                            title="Средний ROI по жанрам",
                            template="plotly_white"
                        )
                        st.plotly_chart(fig_genres_roi, use_container_width=True, key='genre_roi_bar_tab2')
                    else:
                        st.write("Нет данных для расчёта ROI по жанрам.")
                else:
                    st.write("Недостаточно данных для расчёта ROI по жанрам.")
            else:
                st.write("Столбцы 'budget' и/или 'revenue' не найдены в датасете.")
        else:
            st.write("Выберите жанры для анализа.")
    else:
        st.write("Столбец 'genres' не найден в датасете.")


    st.subheader("Анализ по актёрам")
    if 'cast' in movies.columns:
        df_cast = movies.explode('cast').dropna(subset=['cast'])

        actor_counts = df_cast['cast'].value_counts().reset_index()
        actor_counts.columns = ['actor', 'count']
        top_n = 10
        actor_counts_top = actor_counts.head(top_n)

        if not actor_counts_top.empty:
            fig_actors = px.bar(
                actor_counts_top, 
                x='actor', 
                y='count',
                labels={'actor': 'Актёр', 'count': 'Количество фильмов'},
                title=f"Топ {top_n} актёров (по количеству фильмов)",
                template="plotly_white"
            )
            st.plotly_chart(fig_actors, use_container_width=True, key='top_actors_bar_tab2')
        else:
            st.write("Нет данных для построения графика топ актёров.")

        # Средний рейтинг по актёрам
        df_actor_rating = df_cast.groupby('cast')['vote_average'].mean().reset_index()
        df_actor_rating = df_actor_rating.sort_values(by='vote_average', ascending=False).head(top_n)
        if not df_actor_rating.empty:
            fig_actor_rating = px.bar(
                df_actor_rating, 
                x='cast', 
                y='vote_average',
                labels={'cast': 'Актёр', 'vote_average': 'Средний рейтинг'},
                title="Средний рейтинг по актёрам (Топ-10)",
                template="plotly_white"
            )
            st.plotly_chart(fig_actor_rating, use_container_width=True, key='actor_avg_rating_bar_tab2')
        else:
            st.write("Нет данных для расчёта среднего рейтинга по актёрам.")
    else:
        st.write("Столбец 'cast' не найден в датасете.")

   
    st.subheader("Топ-10 фильмов по ROI")
    if 'budget' in df_filtered.columns and 'revenue' in df_filtered.columns:
        df_filtered_roi = df_filtered[
            (df_filtered['budget'] >= 50000) & 
            (df_filtered['revenue'] >= 50000)
        ].copy()
        if not df_filtered_roi.empty:
            df_filtered_roi['ROI'] = (df_filtered_roi['revenue'] - df_filtered_roi['budget']) / df_filtered_roi['budget']
            df_filtered_roi = df_filtered_roi[
                (df_filtered_roi['ROI'] >= -1) & 
                (df_filtered_roi['ROI'] <= 10)
            ]
            top_roi_movies = df_filtered_roi.sort_values(by='ROI', ascending=False).head(10)
            if not top_roi_movies.empty:
                fig_top_roi = px.bar(
                    top_roi_movies, 
                    x='title', 
                    y='ROI',
                    labels={'title': 'Фильм', 'ROI': 'ROI'},
                    title="Топ-10 фильмов по ROI",
                    template="plotly_white"
                )
                st.plotly_chart(fig_top_roi, use_container_width=True, key='top_roi_movies_bar_tab2')
            else:
                st.write("Нет данных для построения графика топ-10 фильмов по ROI.")
        else:
            st.write("Недостаточно данных для расчёта ROI (бюджет и сборы >= 50,000 USD).")
    else:
        st.write("Столбцы 'budget' и/или 'revenue' не найдены в датасете.")
