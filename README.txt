Описание работы программы

1.	CNN, производящая бинарную классификацию лиц (male-female), обучалась на датасете https://www.kaggle.com/cashutosh/gender-classification-dataset, модель бралась из https://github.com/jayrodge/Binary-Image-Classifier-PyTorch с некоторыми модификациями (указаны в коде). Binary_face_classifier.py
2.	Далее для извлечения отдельных лиц использовался pretrained Haar feature-based cascade classifier, классификатор скачивался с https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_alt2.xml (face_recognition.py)
3.	Отдельные лица классифицировались с помощью обученной ранее модели

Для доведения решения до рабочего стоит обучать модель дольше (использовался малопроизводительный компьютер, к тому же с AMD – cuda использовать не получилось, loss function очень далека от стационара), попробовать другие модели CNN (VGG-16?)
