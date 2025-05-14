import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

df = pd.read_csv('notebooks/parsed_data.csv')

st.sidebar.title("Навигация")
section = st.sidebar.radio("Выберите раздел",["Главная", "Данные", "EDA", "Тренды и закономерности", "Выводы"])

# Главная страница
if section == "Главная":
    st.title('Исследование характеристик лауреатов и номинантов премии "Оскар" по данным Кинопоиска')
    st.header("О проекте")
    st.markdown("""
    ### Источник данных
    Данные были собраны с онлайн-кинотеатра "Кинопоиск" и содержат информацию о фильмах с различными атрибутами:
    - Название
    - Наличие статуса победителя премии "Оскар"
    - Год выпуска
    - Жанр
    - Страна выпуска
    - Рейтинг

    ### Возможности приложения
    - Просмотр и фильтрация исходных данных
    - Первичный анализ данных (EDA)
    - Выявление трендов и закономерностей
    - Формирование выводов и рекомендаций
    """)

# Раздел с данными
elif section == "Данные":
    st.header("Исходные данные")

    # Показать весь датафрейм с возможностью фильтрации
    st.subheader("Полный набор данных")
    st.dataframe(df, use_container_width=True)

    # Основные метрики
    st.subheader("Основные метрики")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Всего записей", len(df))
    with col2:
        st.metric("Победителей", df['Победитель'].sum())
    with col3:
        st.metric("Годы охвата", f"{int(df['Год выпуска'].min())}-{int(df['Год выпуска'].max())}")

    # Распределение по категориям
    st.subheader("Распределение по категориям")

    tab1, tab2 = st.tabs(["Жанры", "Страны"])

    with tab1:
        genre_counts = df['Жанр'].value_counts().nlargest(10)
        fig, ax = plt.subplots(figsize=(10, 6))
        genre_counts.plot(kind='bar', ax=ax, color='skyblue')
        ax.set_title('Топ-10 жанров')
        ax.set_xlabel('Жанр')
        ax.set_ylabel('Количество фильмов')
        st.pyplot(fig)

    with tab2:
        country_counts = df['Страна выпуска'].value_counts().nlargest(10)
        fig, ax = plt.subplots(figsize=(10, 6))
        country_counts.plot(kind='bar', ax=ax, color='lightgreen')
        ax.set_title('Топ-10 стран производства')
        ax.set_xlabel('Страна')
        ax.set_ylabel('Количество фильмов')
        st.pyplot(fig)

# Раздел EDA
elif section == "EDA":
    st.header("Первичный анализ данных (EDA)")

    # Типы данных
    st.subheader("Типы данных")
    st.write(df.dtypes)

    # Анализ частотности значений
    st.subheader("Частотный анализ категориальных признаков")

    col1, col2 = st.columns(2)

    with col1:
        st.write("##### Распределение жанров:")
        st.write(df['Жанр'].value_counts())

    with col2:
        st.write("##### Распределение стран выпуска:")
        st.write(df['Страна выпуска'].value_counts())

    # Генерация WordCloud
    st.subheader("Распределение жанров (WordCloud)")
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='#121212',
        colormap='viridis',
        collocations=False,
        min_font_size=10,
        max_words=1000
    ).generate(' '.join(df['Жанр'].astype(str)))

    # Отображение
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.set_facecolor('#121212')
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

    # Анализ рейтинга
    st.subheader("Анализ рейтинга фильмов")
    st.dataframe(df.describe(), use_container_width=True)
    st.write(f"Мода рейтинга: {df['Рейтинг'].mode()[0]:.2f}")
    st.write(f"Средний рейтинг: {df['Рейтинг'].mean():.2f}")
    st.write(f"Медианный рейтинг: {df['Рейтинг'].median():.2f}")

    st.markdown("""
        **Наблюдения:**
        - Среднее и медиана практически совпадают, что говорит об отсутствии сильных выбросов
        - Мода немного выше, что указывает на возможное наличие выбросов
        """)

    # Визуализация выбросов
    st.subheader("Визуализация выбросов в рейтингах")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x=df['Рейтинг'], ax=ax, color='skyblue')
    ax.set_title('Ящик с усами (Boxplot) для выявления выбросов')
    st.pyplot(fig)

    st.markdown("""
        **Анализ boxplot:**
        - Наличие выбросов в области высоких рейтингов подтверждается
        - Основная масса данных сосредоточена в диапазоне примерно 6.5-8.0
        """)

    st.subheader("Пропущенные значения")
    missing = df.isnull().sum().to_frame(name='Количество пропусков')
    st.dataframe(missing, use_container_width=True)

    st.subheader("Распределение рейтингов")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['Рейтинг'], bins=30, kde=True, ax=ax, color='purple')
    ax.set_title('Распределение рейтингов фильмов')
    ax.set_xlabel('Рейтинг')
    ax.set_ylabel('Количество фильмов')
    st.pyplot(fig)

    st.subheader("Рейтинг по жанрам")
    top_genres = df['Жанр'].value_counts().nlargest(5).index
    filtered_df = df[df['Жанр'].isin(top_genres)]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=filtered_df, x='Жанр', y='Рейтинг', ax=ax)
    ax.set_title('Распределение рейтингов по топ-5 жанрам')
    ax.set_xlabel('Жанр')
    ax.set_ylabel('Рейтинг')
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Раздел трендов и закономерностей
elif section == "Тренды и закономерности":
    st.header("Тренды и закономерности")

    # Фильтры
    st.subheader("Фильтры")
    col1, col2 = st.columns(2)

    with col1:
        year_range = st.slider(
            "Диапазон годов",
            min_value=int(df['Год выпуска'].min()),
            max_value=int(df['Год выпуска'].max()),
            value=(int(df['Год выпуска'].min()), int(df['Год выпуска'].max()))
        )

    with col2:
        selected_genres = st.multiselect(
            "Выберите жанры",
            options=df['Жанр'].unique(),
            default=df['Жанр'].value_counts().nlargest(3).index.tolist()
        )

    # Применение фильтров
    filtered_df = df[
        (df['Год выпуска'] >= year_range[0]) &
        (df['Год выпуска'] <= year_range[1])
        ]

    if selected_genres:
        filtered_df = filtered_df[filtered_df['Жанр'].isin(selected_genres)]

    # Визуализации
    st.subheader("Динамика выпуска фильмов по годам")
    fig, ax = plt.subplots(figsize=(12, 6))
    movies_per_year = filtered_df['Год выпуска'].value_counts().sort_index()
    movies_per_year.plot(kind='line', marker='o', ax=ax, color='darkblue')
    ax.set_title('Количество фильмов по годам')
    ax.set_xlabel('Год')
    ax.set_ylabel('Количество фильмов')
    st.pyplot(fig)

    st.subheader("Средний рейтинг по годам")
    fig, ax = plt.subplots(figsize=(12, 6))
    rating_per_year = filtered_df.groupby('Год выпуска')['Рейтинг'].mean()
    rating_per_year.plot(kind='line', marker='o', ax=ax, color='darkred')
    ax.set_title('Средний рейтинг фильмов по годам')
    ax.set_xlabel('Год')
    ax.set_ylabel('Средний рейтинг')
    st.pyplot(fig)

    st.subheader("Топ фильмов по рейтингу")
    top_movies = filtered_df.sort_values('Рейтинг', ascending=False).head(10)
    st.dataframe(top_movies[['Название', 'Год выпуска', 'Жанр', 'Рейтинг']],
                 use_container_width=True)

# Раздел выводов
elif section == "Выводы":
    st.header("Выводы")

    st.subheader("Ключевые инсайты")
    st.markdown("""
    1. **Распределение жанров**: Драма является самым популярным жанром в наборе данных, но при этом жанр Биографии имеет в среднем рейтинг выше других.
    2. **Рейтинги**: Фильмы-победители премий имеют средний рейтинг 7.4, тогда как непобедители — 7.2. При этом фильмы "Побег из Шоушенка" и "Зеленая миля" лауреатами не являются, но обладают наибольши рейтингом среди всех даннхы - 9.1. Если причиной отсутствия "Оскара" у "Зеленой мили" может быть наличие мистических элементов и тюремная тематика, которые несмотря на глубокий драматизм в сюжете, традиционно не пользуются расположением членов Американской киноакадемии, то "Побег из Шоушенка" вышел в год, который был очень богат на удачные фильмы ("Криминальное чтиво", "Леон" и др.), и награды досталась такой картине, как «Форрест Гамп».
    3. **Ведущая страна**: Страна с наибольшим количеством фильмов - США.
    4. **Тренды**: Топ-3 самых высоко оцененных фильмов обладают жанром Драма и выпущены в 1990-х годах.
    5. **Интересное примечание**: Отсутствие "синдрома сиквела": В 1990-х было меньше франшиз, потому что оригинальные сценарии ценились выше.
    """)

    st.subheader("Как можно дополнить наше исследование в будущем?")
    st.markdown("""
    1. **Финансовая часть**: Для более глубокого анализа было бы полезно добавить информацию о бюджетах фильмов и кассовых сборах.
    3. **Жанровый анализ**: Интересно было бы изучить, как менялась популярность разных жанров с течением времени.
    4. **Сравнение стран**: Можно углубиться в сравнение кинематографа разных стран.
    """)
