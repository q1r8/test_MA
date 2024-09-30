import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F

import cv2
from PIL import Image

#Класс модели
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        self.cnn1 = nn.Sequential(
            
            nn.Conv2d(3, 96, kernel_size=11,stride=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            
            nn.Conv2d(96, 256, kernel_size=5,stride=1,padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout2d(p=0.3),

            nn.Conv2d(256,384 , kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(384,256 , kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout2d(p=0.3),

        )
        
        self.fc1 = nn.Sequential(
            nn.Linear(30976, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
            
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
            
            nn.Linear(128,16))
        
        
  
    def forward_once(self, x): 
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


#если запускаем на CPU:
model = torch.load('путь_до_весов_модели', map_location=torch.device('cpu'))
# если запускаем на GPU:
# model = torch.load('путь до весов').cuda()


#Препроцессинг изображения на входе
def preprocess_img(img_path, device='cpu'):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = transforms.Resize((105, 105))(img)
    img = transforms.ToTensor()(img).unsqueeze(0)
    return img.to(device)


def image_similarity(img1_path, img2_path):
    #Считываем и предобрабатываем изображения
    img1 = preprocess_img(img1_path)
    img2 = preprocess_img(img2_path)
    
    #получаем их эмбеддинги
    with torch.no_grad():
        output1, output2 = model(img1, img2)
    #считаем евклидово расстояние между ними
    eucledian_distance = F.pairwise_distance(output1, output2).cpu().detach().numpy()[0]
    #если оно больше 1 возвращаем ноль (изображения совсем не похожи)
    if eucledian_distance >= 1:
        return 0
    #иначе возвращаем 1 - дистанцию (схожесть)
    return 1 - eucledian_distance

#Запуск модели, посчет схожести изображений (сюда ввести два пути до изображений)
if __name__ == '__main__':
    similarity = image_similarity('путь_до_изображения1', 'путь_до_изображения2')
    print(round(similarity, 3))



