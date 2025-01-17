[<img src="https://img.shields.io/badge/Streamlit-%40movies_recsys_analytics-green">](https://huggingface.co/spaces/HounchPounchGit/MoviesRecSysStreamlit)

# Movie Recommender System and Analytics Dashboard
## Описание
В данном репозитории представлена [рекомендательная система](https://huggingface.co/spaces/SvetlanaErmakova/movies_recsys_analytics), которая подбирает наиболее релеватные фильмы для пользователя по запросу, содержащему название понравившегося фильма, при помощи машинного обучения. Рекомендации строятся изсходя из косинусных близостей `tf-idf` векторов, составленных на основе выделенных признаков фильмов, таких как описание, режиссер и т.д. Так же для более продвинутых рекомендаций дополнительно используется ранжирование на основе средневзвешенного рейтинга. 

Предоставлена возможность получать краткую аналитику и описание по рекомендованным фильмам. Реализован общий дашборд для аналитики всего датасета в отдельной вкладке.


<p align="center">
  <img src="screenshots/1.png" height="500" alt="Ray Image">
</p>



<p align="center">
  <img src="screenshots/2.png" height="500" alt="Ray Image">
</p>



<p align="center">
  <img src="screenshots/3.png" height="500" alt="Ray Image">
</p>

## Установка
1. Клонируйте репозиторий;
2. Скопируйте `env_example` под именем `.env`, откройте его и заполните переменные;
3. Установите соответвующие модули из файла `requirements.txt`;
4. Запустите приложение командой `streamlit run app.py`;

