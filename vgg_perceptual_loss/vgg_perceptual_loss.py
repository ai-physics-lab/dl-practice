import torch
import torchvision
import torchvision.models as models

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, path='/content/drive/MyDrive/ai-physics-lab/VGGPerceptualLoss/vgg16.pth', resize=True):
        super(VGGPerceptualLoss, self).__init__()
        # load trained model
        # load vgg into cuda if available, load into cpu if cuda not available
        if torch.cuda.is_available():
            self.vgg = models.vgg16().to('cuda')
        else:
            self.vgg = models.vgg16().to('cpu')
        self.vgg.load_state_dict((torch.load(path)))
        self.vgg.eval()
        # preparation to get feature maps from the middle of the model
        blocks = [
            self.vgg.features[:4],
            self.vgg.features[4:9],
            self.vgg.features[9:16],
            self.vgg.features[16:23]
        ]
        # No grad to prevent back propagation
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        # Combine the blocks into ModuleList
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                # add loss by using the feature map in the middle of the model
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                # add loss by using the "style layer" calculated from the feature maps
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss

class VGGPerceptualFeatureMaps(torch.nn.Module):
    def __init__(self, path='/content/drive/MyDrive/ai-physics-lab/VGGPerceptualLoss/vgg16.pth', resize=True):
        super(VGGPerceptualFeatureMaps, self).__init__()
        # INIT 1. Load Model
        # load vgg into cuda if available, load into cpu if cuda not available
        if torch.cuda.is_available():
            self.vgg = models.vgg16().to('cuda')
        else:
            self.vgg = models.vgg16().to('cpu')
        self.vgg.load_state_dict((torch.load(path)))
        # use model in evaluation mode
        self.vgg.eval()

        # Init 2. Prep. feature maps
        # 2-1 preparation to get feature maps from the middle of the model
        blocks = [
            self.vgg.features[:4],
            self.vgg.features[4:9],
            self.vgg.features[9:16],
            self.vgg.features[16:23],
            self.vgg.features[23:30]
        ]
        
        # 2-2 No grad to prevent back propagation
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        # 2-3 Combine the blocks into ModuleList
        self.blocks = torch.nn.ModuleList(blocks)

        # Init 3. Others
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        input_features = []
        target_features = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                # add featuremaps
                input_features.append(x)
                target_features.append(y)
        return input_features, target_features