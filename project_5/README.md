# PROJECT-5. Решение задачи регрессии.

## Оглавление
[1. Цели и суть проекта](#Цели-и-суть-проекта)

[2. Этапы проекта](#Этапы-проекта)

[3. “Первичная обработка данных”](#Первичная-обработка-данных)

[4. “Разведывательный анализ данных”](#Разведывательный-анализ-данных)

[5. “Отбор и преобразование признаков”](#Отбор-и-преобразование-признаков)

[6. “Решение задачи регрессии: линейная регрессия и деревья решений”](#Решение-задачи-регрессии-линейная-регрессия-и-деревья-решений)

[7. “Решение задачи регрессии: ансамбли моделей и построение прогноза”](#Решение-задачи-регрессии-ансамбли-моделей-и-построение-прогноза)

[8. Выводы](#Выводы)

[9. Заключение](#Заключение)

[10. Ресурсы проекта](#Ресурсы-проекта)


-----
### **Цели и суть проекта**

*Проблема*:

В данном модуле вам предстоит решить настоящую задачу машинного обучения, направленную на автоматизацию бизнес-процессов. Мы построим модель, которая будет предсказывать общую продолжительность поездки на такси в Нью-Йорке.


Представьте, что вы заказываете такси из одной точки Нью-Йорка в другую, причём необязательно, что конечная точка должна находиться в пределах города. Сколько вы должны будете заплатить за поездку?

Известно, что стоимость такси в США рассчитывается на основе фиксированной ставки и тарифной стоимости, величина которой зависит от времени и расстояния. Тарифы варьируются в зависимости от города.

В свою очередь, время поездки зависит от множества факторов, таких как направление поездки, время суток, погодные условия и так далее.

Таким образом, если мы разработаем алгоритм, способный определять длительность поездки, мы сможем прогнозировать её стоимость самым тривиальным образом, например, просто умножая стоимость на заданный тариф.

Сервисы такси хранят огромные объёмы информации о поездках, включая такие данные, как конечная и начальная точки маршрута, дата поездки и её продолжительность. Эти данные можно использовать для того, чтобы прогнозировать длительность поездки в автоматическом режиме с привлечением искусственного интеллекта.


*Бизнес-задача*: определить характеристики и с их помощью спрогнозировать длительность поездки на такси.

*Техническая задача для специалиста в Data Science*: построить модель машинного обучения, которая на основе предложенных характеристик клиента будет предсказывать числовой признак — время поездки такси, то есть решить задачу регрессии.


*Основные цели*:
       
- Сформировать набор данных на основе нескольких источников информации.
- Спроектировать новые признаки с помощью Feature Engineering и выявить наиболее значимые при построении модели.
- Исследовать предоставленные данные и выявить закономерности.
- Построить несколько моделей и выбрать из них ту, которая показывает наилучший результат по заданной метрике.
- Спроектировать процесс предсказания длительности поездки для новых данных.
- Загрузить своё решение на платформу Kaggle, тем самым поучаствовав в настоящем Data Science-соревновании.

Во время выполнения проекта вы отработаете навыки работы с несколькими источниками данных, генерации признаков, разведывательного анализа и визуализации данных, отбора признаков и, конечно же, построения моделей машинного обучения.


### **Этапы проекта:**

*Проект будет состоять из пяти частей:*

1. Первичная обработка данных.
В рамках этой части вам предстоит сформировать набор данных на основе предложенных нами источников информации, а также обработать пропуски и выбросы в данных.

2. Разведывательный анализ данных (EDA).
Вам необходимо будет исследовать данные, нащупать первые закономерности и выдвинуть гипотезы.

3. Отбор и преобразование признаков.
На этом этапе вы перекодируете и преобразуете данные таким образом, чтобы их можно было использовать при решении задачи регрессии — для построения модели.

4. Решение задачи регрессии: линейная регрессия и деревья решений.
На данном этапе вы построите свои первые прогностические модели и оцените их качество. Тем самым вы создадите так называемый baseline, который поможет вам ответить на вопрос: «Решаема ли вообще представленная задача?»

5. Решение задачи регрессии: ансамбли моделей и построение прогноза.
На заключительном этапе вы сможете доработать своё предсказание с использованием более сложных алгоритмов и оценить, с помощью какой модели возможно сделать более качественные прогнозы.

Важно, что иногда задания не несут практической ценности (т. е. какие-то из показателей не нужно было бы рассчитывать при решении задачи в реальной ситуации), однако они отлично помогают идентифицировать правильность выполнения вами необходимых действий.


---
### **Первичная обработка данных**

Вам будет предоставлен набор данных, содержащий информацию о поездках на жёлтом такси в Нью-Йорке за 2016 год. Первоначально данные были выпущены Комиссией по Такси и Лимузинам Нью-Йорка и включают в себя информацию о времени поездки, географических координатах, количестве пассажиров и несколько других переменных.

На этом этапе необходимо поближе познакомиться с данными: 

- посмотреть структуру таблицы, 
- ее содержимое, 
- количество строк,
- количество пропусков,
- поискать дубликаты,
- обработать выбросы.

Перед началом работы необходимо подгрузить данные и убедиться, что все прошло успешно.

Есть данные о почти 1.5 миллионах поездок и 11 характеристиках, которые описывают каждую из поездок.

Данные о клиенте и таксопарке:

    id - уникальный идентификатор поездки
    vendor_id - уникальный идентификатор поставщика (таксопарка), связанного с записью поездки

Временные характеристики:

    pickup_datetime - дата и время, когда был включен счетчик поездки
    dropoff_datetime - дата и время, когда счетчик был отключен

Географическая информация:

    pickup_longitude - долгота, на которой был включен счетчик
    pickup_latitude - широта, на которой был включен счетчик
    dropoff_longitude - долгота, на которой счетчик был отключен
    dropoff_latitude - широта, на которой счетчик был отключен

Прочие признаки:

    passenger_count - количество пассажиров в транспортном средстве (введенное водителем значение)
    store_and_fwd_flag - флаг, который указывает, сохранилась ли запись о поездке в памяти транспортного средства перед отправкой поставщику. Y - хранить и пересылать, N - не хранить и не пересылать поездку.

Целевой признак:

    trip_duration - продолжительность поездки в секундах

Начнём наше исследование со знакомства с предоставленными данными. Также мы подгрузим дополнительные источники данных и расширим исходный датасет.


### **Разведывательный анализ данных**

В этой части проекта мы:

- исследуем сформированный набор данных; 
- попробуем найти закономерности, позволяющие сформулировать предварительные гипотезы относительно того, какие факторы являются решающими в определении длительности поездки;
- дополним наш анализ визуализациями, иллюстрирующими исследование (постарайтесь оформлять диаграммы с душой, а не «для галочки»: навыки визуализации полученных выводов обязательно пригодятся вам в будущем).


### **Отбор и преобразование признаков**

Перед тем как перейти к построению модели, осталось сделать ещё несколько шагов.

Следует помнить, что многие алгоритмы машинного обучения не могут обрабатывать категориальные признаки в их обычном виде. Поэтому нам необходимо их закодировать.

Надо отобрать признаки, которые мы будем использовать для обучения модели.

Необходимо масштабировать и трансформировать некоторые признаки для того, чтобы улучшить сходимость моделей, в основе которых лежат численные методы.



### **Решение задачи регрессии: линейная регрессия и деревья решений**

На этом этапе мы решим задачу регрессии: отберем признаки, обучим модель, сделаем прогноз и оценим его качество.

На данном этапе построим свои первые прогностические модели и оценим их качество. Тем самым создадим так называемый baseline, который поможет ответить на вопрос: «Решаема ли вообще представленная задача?»

### **Решение задачи регрессии: ансамбли моделей и построение прогноза**

Мы уже смогли обучить несложные модели, и теперь пришло время усложнить их, а также посмотреть, улучшится ли результат (если да, то насколько). 
На заключительном этапе сможем доработать своё предсказание с использованием более сложных алгоритмов и оценить, с помощью какой модели возможно сделать более качественные прогнозы.


### **Выводы**

Решена задача регрессии, с подобранной оптимальной моделью и лучшей метрикой.


----
### **Заключение**

Вы справились с настоящим проектом, решив достаточно сложную, но важную и актуальную задачу. Теперь вы можете решить полноценную задачу регрессии, начиная от предобработки данных и заканчивая оценкой качества построенных моделей и отбора наиболее значимых факторов.

Не останавливайтесь на полученном решении этой задачи. Это лишь один из возможных вариантов. Вы можете попробовать улучшить качество предсказания, используя более продвинутые подходы для генерации признаков, обработки пропусков, поиска выбросов, отбора признаков и так далее. 

Поэкспериментируйте с методами оптимизации гиперпараметров алгоритмов. Но будьте осторожны, так как в обучающем наборе очень много данных и подбор внешних параметров может занимать много времени. Выбирайте диапазоны оптимизации с умом.

Вы также можете воспользоваться более сложными методами машинного обучения, например современными вариантами бустинга, такими как CatBoost от Яндекса или LightGBM от Microsoft.

Можно даже использовать стекинг, агрегировав несколько мощных моделей в одну.

---
### **Ресурсы проекта**

Исходные таблицы, используемые в проекте, находятся здесь:

https://drive.google.com/drive/folders/1cM2hKKl4F1ecKtjKbROV6Lb4Gl336MMz





:arrow_up: [к оглавлению](https://github.com/TatyanaTmf/ds_game/tree/main/project_5/README.md#Оглавление)