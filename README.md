[<img src="https://img.shields.io/badge/Streamlit-%40RecSysApp-green">](https://huggingface.co/spaces/HounchPounchGit/MoviesRecSysStreamlit)

# Movie Recommender System
## Описание
В данном репозитории представлена [рекомендательная система](https://huggingface.co/spaces/HounchPounchGit/MoviesRecSysStreamlit), которая подбирает наиболее релеватные фильмы для пользователя по запросу, содержащему название понравившегося фильма, при помощи машинного обучения. Рекомендации строятся изсходя из косинусных близостей `tf-idf` векторов, составленных на основе выделенных признаков фильмов, таких как описание, режиссер и т.д. Так же для более продвинутых рекомендаций дополнительно используется ранжирование на основе средневзвешенного рейтинга.


<p align="center">
  <img src="screenshots/app1.png" height="500" alt="Ray Image">
</p>
Написано streamlit приложение, упакованное в Docker и задеплоенное в облако. Ноутбук с подробным обучением находится в репозитории.

## Установка
### Docker + Docker Compose
1. Возьмите файл `docker-compose.yml` из репозитория;
2. Возьмите файл `env_example` там же, переименуйте как `.env`, откройте и заполните переменные;
3. Запустите приложение: `docker compose up -d` (или `docker-compose up -d` на старых версиях Docker);
4. Проверьте, что контейнер поднялся: `docker compose ps`
5. Длаее следуйте инструкции, выведенной после команды `docker-compose logs`
