Веса модели - https://drive.google.com/file/d/1PmwWxg2BB-y8z1SFkBc5eRQ6MmiNOqu4/view?usp=sharing

csv для подсчета метрик - https://drive.google.com/file/d/1VxglHIa-l9iztLD53is7nuiwoPF9fqjL/view?usp=sharing
## Подход

Популярным решением этой задачи являются всевозможные подходы связанные с metric learning. Я остановился на подходе саимской сети, его основной смысл заключается в том, что у нас есть "две" нейронных сети с одинаковыми весами, через которые мы прогоняем два изображения на входе и на выходе имеем два эмбеддинга, на основе которых считаем расстояние между двумя изображениями.
![telegram-cloud-photo-size-2-5465130625150348536-y](https://github.com/user-attachments/assets/7365e80d-4e3c-44e4-b90e-7d007a275ad0)

## Данные

Предварительно изучил данные и намайнил пары изображений. По итогу в трейн сете у меня было ~7k сетов, ~3,5k позитивных (пара изображений принадлежат одному классу) и ~3,5k негативных (пара изображений из разных классов)

В качестве feature extactor использовал 4 сверточных слоя с разными размерами ядер, поверх них сделал 3 full connected слоя для понижения размерности эмбеддинга и классификации

Обучал 20 эпох с contastive loss.

## Метрики
Считал основные метрики для классификации с разными отсечками на 1000 семплах пар, не входящих в обучение (500 пар позитивных, 500 негативных):
Если значение расстояние меньше treshold'а - целевой класс, больше - негативный

treshold - 0.05

<img width="373" alt="изображение" src="https://github.com/user-attachments/assets/63551500-fd08-4057-9996-47921c3ea0fe">

treshold - 0.1

<img width="371" alt="изображение" src="https://github.com/user-attachments/assets/b84ffdbb-9d0b-4692-b961-6fd9c6e58850">

treshold - 0.15

<img width="373" alt="изображение" src="https://github.com/user-attachments/assets/b71147d7-04f8-41ca-9a19-04ef79c6d687">

treshold - 0.2

<img width="379" alt="изображение" src="https://github.com/user-attachments/assets/fecdc6b5-5849-423a-8848-e05d5ca7a794">


## Развитие
* Очень много зависит от того, как нагенерить данные, поэтому в первую очередь бы уделил этому много времени. Составил бы хард семплы, когда изображения очень похожи, но принадлежат разным классам и просто более детально собирал бы остальные пары
* Попробовал бы различные metric learning подходы:
  * Triplet loss; лосс в котором обучение строится на тройках изображений: основное, изображение того же класса и изображение другого класса
  * ArcFace; лосс в котором мы обучаем модель, как обычный классификатор, но сильнее разделяем разные классы друг от друга на гиперплоскости
* Экспериментировал бы с архитектурами энкодеров, попробывал бы маленькие сверточные сети по типу mobilenet, effnet (на больших архитектурах модель быстро переобучается). Различные  visual трансформеры, по типу DeiT.
* Сделал бы классификацию эмбеддингов, вместо посчета расстояния (PCA (если эмбеддинг большой) + линейная модель классификации (XGBoost)) 
  
