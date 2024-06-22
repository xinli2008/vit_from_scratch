from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Resize
import torchvision.transforms
from torchvision.transforms import ToPILImage

class my_dataset(Dataset):
    def __init__(self, is_train = True):
        super(my_dataset, self).__init__()
        self.dataset = torchvision.datasets.MNIST(root="./mnist/", train = is_train, download = True)
        self.image_convert = torchvision.transforms.Compose([
            ToTensor(),
            Resize(224, antialias=True)
        ])

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        image, label = self.dataset[index]
        return self.image_convert(image), label

if __name__ == "__main__":
    my_dataset = my_dataset()
    image, label = my_dataset[0]    
    image_pil = ToPILImage()(image).save("./image.png")
