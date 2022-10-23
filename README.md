# GANs
Некоторое время потратил на изучение генеративно-состязательных сетей, в репозиториях пару примеров архитектур, написанных по статьям с некоторыми дополнениями: 

1) Fc_GAn - простой для понимания набор из двух полносвязных сетей (генератора и дискриминатора), обученный на мнисте. Для понимания смысла работы такого рода сетей самое оно. 

2) DCGAN - основное отличие данного гана от предыдущего в использовании сверточных и transpose слоев взамен полносвязных в скрытом слое. Прикольно(:

3) WGAN - целью ганов с точки зрениия математики как правило является повторение генератором распределения вероятности реальных данных. Для оценки "похожести" этих распределений в них как правило используют дивергенцию Йенса-Шеннона. Но есть проблема - данный оператор обладает проблемами с градиентом, что для генеративно-состязательных сетей особенно чувствиетльно, поэтому в данном типе сетей используется альтернативный подход к сравнению - расстояние Вассерштайна (полезные ссылки: https://www.alexirpan.com/2017/02/22/wasserstein-gan.html https://arxiv.org/abs/1701.07875). 

4) SRGAN - пока просто посторил статью и примерно разобрался, как он работает, в планах оптимизировать его работу для больших картинок и завернуть готовую модель в бота в тг просто потому что круто (https://arxiv.org/abs/1809.00219). 

5) в будущем в планах также попробовать StyleGAN, pix2pix, Stable diffusion через опенвино на компе потестил, прикольно, только не очень ясно, как оно работает после текстового трансформера. 

Ну и в конце как обычно скример![2022-10-23 14 31 05](https://user-images.githubusercontent.com/90149954/197389677-383702b4-f36b-4153-8f4a-feaad96310ac.jpg)
