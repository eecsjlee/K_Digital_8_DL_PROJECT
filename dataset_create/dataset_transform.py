import torchvision.transforms as transforms
import torchvision.transforms.functional as F

class FixedRotation(object):
    def __init__(self, angle, fill = (127,127,127)):
        self.angle = angle
        self.fill = fill
    def __call__(self, img):
        return F.rotate(img, self.angle, expand=True, fill= self.fill)

transform_low_res = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ColorJitter(brightness=(1.0, 4.0), contrast=(1.0, 4.0), saturation=(1.0, 4.0), hue=(0.1, 0.5)),
    FixedRotation(angle=180, fill = (127,127,127)),
    transforms.ToTensor()
])