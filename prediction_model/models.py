
model_list = [
    'vit_b_16',
    'vit_b_32',
    'vit_l_16',
    'vit_l_32',
    'vit_h_14',
    'swin_t',
    'swin_s',
    'swin_b',
    # 'swin_v2_t', # torchvision>=0.14
    # 'swin_v2_s', # torchvision>=0.14
    # 'swin_v2_b', # torchvision>=0.14
    'poolformer_s12',
    'resnet18',
    'resnet34',
    'resnet50',
    'resnet101',
    'resnet152',
    'vgg11',
    'vgg11_bn',
    'vgg13',
    'vgg13_bn',
    'vgg16',
    'vgg16_bn',
    'vgg19',
    'vgg19_bn',
]

type_list = [
    'transfer_learning',
    'fine_tuning',
    'normal',
]

def model(modelname, learningtype, path_to_model=None):
    assert modelname in model_list, NotImplementedError(f'Model name "{modelname}" is unknown. Please select any of {model_list}')
    assert learningtype in type_list, NotImplementedError(f'Learning type "{learningtype}" is unknown. Please select any of {type_list}')
    # vit=========================================================================
    if modelname=='vit_b_16':
        from torchvision.models import vit_b_16, ViT_B_16_Weights
        if learningtype=='transfer_learning':
            model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
            for param in model.parameters():
                param.requires_grad = False
        elif learningtype=='fine_tuning':
            model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        elif learningtype=='normal':
            model = vit_b_16(weights=None)
        return model
    elif modelname=='vit_b_32':
        from torchvision.models import vit_b_32, ViT_B_32_Weights
        if learningtype=='transfer_learning':
            model = vit_b_32(weights=ViT_B_32_Weights.DEFAULT)
            for param in model.parameters():
                param.requires_grad = False
        elif learningtype=='fine_tuning':
            model = vit_b_32(weights=ViT_B_32_Weights.DEFAULT)
        elif learningtype=='normal':
            model = vit_b_32(weights=None)
        return model
    elif modelname=='vit_l_16':
        from torchvision.models import vit_l_16, ViT_L_16_Weights
        if learningtype=='transfer_learning':
            model = vit_l_16(weights=ViT_L_16_Weights.DEFAULT)
            for param in model.parameters():
                param.requires_grad = False
        elif learningtype=='fine_tuning':
            model = vit_l_16(weights=ViT_L_16_Weights.DEFAULT)
        elif learningtype=='normal':
            model = vit_l_16(weights=None)
        return model
    elif modelname=='vit_l_32':
        from torchvision.models import vit_l_32, ViT_L_32_Weights
        if learningtype=='transfer_learning':
            model = vit_l_32(weights=ViT_L_32_Weights.DEFAULT)
            for param in model.parameters():
                param.requires_grad = False
        elif learningtype=='fine_tuning':
            model = vit_l_32(weights=ViT_L_32_Weights.DEFAULT)
        elif learningtype=='normal':
            model = vit_l_32(weights=None)
        return model
    elif modelname=='vit_h_14':
        from torchvision.models import vit_h_14, ViT_H_14_Weights
        if learningtype=='transfer_learning':
            model = vit_h_14(weights=ViT_H_14_Weights.IMAGENET1K_SWAG_LINEAR_V1)
            for param in model.parameters():
                param.requires_grad = False
        elif learningtype=='fine_tuning':
            model = vit_h_14(weights=ViT_H_14_Weights.IMAGENET1K_SWAG_LINEAR_V1)
        elif learningtype=='normal':
            model = vit_h_14(weights=None)
        return model
    # swin=========================================================================
    elif modelname=='swin_t':
        from torchvision.models import swin_t, Swin_T_Weights
        if learningtype=='transfer_learning':
            model = swin_t(weights=Swin_T_Weights.DEFAULT)
            for param in model.parameters():
                param.requires_grad = False
        elif learningtype=='fine_tuning':
            model = swin_t(weights=Swin_T_Weights.DEFAULT)
        elif learningtype=='normal':
            model = swin_t(weights=None)
        return model
    elif modelname=='swin_s':
        from torchvision.models import swin_s, Swin_S_Weights
        if learningtype=='transfer_learning':
            model = swin_s(weights=Swin_S_Weights.DEFAULT)
            for param in model.parameters():
                param.requires_grad = False
        elif learningtype=='fine_tuning':
            model = swin_s(weights=Swin_S_Weights.DEFAULT)
        elif learningtype=='normal':
            model = swin_s(weights=None)
        return model
    elif modelname=='swin_b':
        from torchvision.models import swin_b, Swin_B_Weights
        if learningtype=='transfer_learning':
            model = swin_b(weights=Swin_B_Weights.DEFAULT)
            for param in model.parameters():
                param.requires_grad = False
        elif learningtype=='fine_tuning':
            model = swin_b(weights=Swin_B_Weights.DEFAULT)
        elif learningtype=='normal':
            model = swin_b(weights=None)
        return model
    elif modelname=='swin_v2_t':
        from torchvision.models import swin_v2_t, Swin_V2_T_Weights
        if learningtype=='transfer_learning':
            model = swin_v2_t(weights=Swin_V2_T_Weights.DEFAULT)
            for param in model.parameters():
                param.requires_grad = False
        elif learningtype=='fine_tuning':
            model = swin_v2_t(weights=Swin_V2_T_Weights.DEFAULT)
        elif learningtype=='normal':
            model = swin_v2_t(weights=None)
        return model
    elif modelname=='swin_v2_s':
        from torchvision.models import swin_v2_s, Swin_V2_S_Weights
        if learningtype=='transfer_learning':
            model = swin_v2_s(weights=Swin_V2_S_Weights.DEFAULT)
            for param in model.parameters():
                param.requires_grad = False
        elif learningtype=='fine_tuning':
            model = swin_v2_s(weights=Swin_V2_S_Weights.DEFAULT)
        elif learningtype=='normal':
            model = swin_v2_s(weights=None)
        return model
    elif modelname=='swin_v2_b':
        from torchvision.models import swin_v2_b, Swin_V2_B_Weights
        if learningtype=='transfer_learning':
            model = swin_v2_b(weights=Swin_V2_B_Weights.DEFAULT)
            for param in model.parameters():
                param.requires_grad = False
        elif learningtype=='fine_tuning':
            model = swin_v2_b(weights=Swin_V2_B_Weights.DEFAULT)
        elif learningtype=='normal':
            model = swin_v2_b(weights=None)
        return model
    # poolformer=========================================================================
    elif modelname=='poolformer_s12':
        import models_poolformer
        from timm.models import load_checkpoint
        model = models_poolformer.poolformer_s12()
        if learningtype=='transfer_learning':
            load_checkpoint(model=model, checkpoint_path=path_to_model)
            for param in model.parameters():
                param.requires_grad = False
        elif learningtype=='fine_tuning':
            load_checkpoint(model=model, checkpoint_path=path_to_model)
        elif learningtype=='normal':
            pass
        return model
    # resnet=========================================================================
    elif modelname=='resnet18':
        from torchvision.models import resnet18, ResNet18_Weights
        if learningtype=='transfer_learning':
            model = resnet18(weights=ResNet18_Weights.DEFAULT)
            for param in model.parameters():
                param.requires_grad = False
        elif learningtype=='fine_tuning':
            model = resnet18(weights=ResNet18_Weights.DEFAULT)
        elif learningtype=='normal':
            model = resnet18(weights=None)
        return model
    elif modelname=='resnet34':
        from torchvision.models import resnet34, ResNet34_Weights
        if learningtype=='transfer_learning':
            model = resnet34(weights=ResNet34_Weights.DEFAULT)
            for param in model.parameters():
                param.requires_grad = False
        elif learningtype=='fine_tuning':
            model = resnet34(weights=ResNet34_Weights.DEFAULT)
        elif learningtype=='normal':
            model = resnet34(weights=None)
        return model
    elif modelname=='resnet50':
        from torchvision.models import resnet50, ResNet50_Weights
        if learningtype=='transfer_learning':
            model = resnet50(weights=ResNet50_Weights.DEFAULT)
            for param in model.parameters():
                param.requires_grad = False
        elif learningtype=='fine_tuning':
            model = resnet50(weights=ResNet50_Weights.DEFAULT)
        elif learningtype=='normal':
            model = resnet50(weights=None)
        return model
    elif modelname=='resnet101':
        from torchvision.models import resnet101, ResNet101_Weights
        if learningtype=='transfer_learning':
            model = resnet101(weights=ResNet101_Weights.DEFAULT)
            for param in model.parameters():
                param.requires_grad = False
        elif learningtype=='fine_tuning':
            model = resnet101(weights=ResNet101_Weights.DEFAULT)
        elif learningtype=='normal':
            model = resnet101(weights=None)
        return model
    elif modelname=='resnet152':
        from torchvision.models import resnet152, ResNet152_Weights
        if learningtype=='transfer_learning':
            model = resnet152(weights=ResNet152_Weights.DEFAULT)
            for param in model.parameters():
                param.requires_grad = False
        elif learningtype=='fine_tuning':
            model = resnet152(weights=ResNet152_Weights.DEFAULT)
        elif learningtype=='normal':
            model = resnet152(weights=None)
        return model
    # vgg=========================================================================
    elif modelname=='vgg11':
        from torchvision.models import vgg11, VGG11_Weights
        if learningtype=='transfer_learning':
            model = vgg11(weights=VGG11_Weights.DEFAULT)
            for param in model.parameters():
                param.requires_grad = False
        elif learningtype=='fine_tuning':
            model = vgg11(weights=VGG11_Weights.DEFAULT)
        elif learningtype=='normal':
            model = vgg11(weights=None)
        return model
    elif modelname=='vgg11_bn':
        from torchvision.models import vgg11_bn, VGG11_BN_Weights
        if learningtype=='transfer_learning':
            model = vgg11_bn(weights=VGG11_BN_Weights.DEFAULT)
            for param in model.parameters():
                param.requires_grad = False
        elif learningtype=='fine_tuning':
            model = vgg11_bn(weights=VGG11_BN_Weights.DEFAULT)
        elif learningtype=='normal':
            model = vgg11_bn(weights=None)
        return model
    elif modelname=='vgg13':
        from torchvision.models import vgg13, VGG13_Weights
        if learningtype=='transfer_learning':
            model = vgg13(weights=VGG13_Weights.DEFAULT)
            for param in model.parameters():
                param.requires_grad = False
        elif learningtype=='fine_tuning':
            model = vgg13(weights=VGG13_Weights.DEFAULT)
        elif learningtype=='normal':
            model = vgg13(weights=None)
        return model
    elif modelname=='vgg13_bn':
        from torchvision.models import vgg13_bn, VGG13_BN_Weights
        if learningtype=='transfer_learning':
            model = vgg13_bn(weights=VGG13_BN_Weights.DEFAULT)
            for param in model.parameters():
                param.requires_grad = False
        elif learningtype=='fine_tuning':
            model = vgg13_bn(weights=VGG13_BN_Weights.DEFAULT)
        elif learningtype=='normal':
            model = vgg13_bn(weights=None)
        return model
    elif modelname=='vgg16':
        from torchvision.models import vgg16, VGG16_Weights
        if learningtype=='transfer_learning':
            model = vgg16(weights=VGG16_Weights.DEFAULT)
            for param in model.parameters():
                param.requires_grad = False
        elif learningtype=='fine_tuning':
            model = vgg16(weights=VGG16_Weights.DEFAULT)
        elif learningtype=='normal':
            model = vgg16(weights=None)
        return model
    elif modelname=='vgg16_bn':
        from torchvision.models import vgg16_bn, VGG16_BN_Weights
        if learningtype=='transfer_learning':
            model = vgg16_bn(weights=VGG16_BN_Weights.DEFAULT)
            for param in model.parameters():
                param.requires_grad = False
        elif learningtype=='fine_tuning':
            model = vgg16_bn(weights=VGG16_BN_Weights.DEFAULT)
        elif learningtype=='normal':
            model = vgg16_bn(weights=None)
        return model
    elif modelname=='vgg19':
        from torchvision.models import vgg19, VGG19_Weights
        if learningtype=='transfer_learning':
            model = vgg19(weights=VGG19_Weights.DEFAULT)
            for param in model.parameters():
                param.requires_grad = False
        elif learningtype=='fine_tuning':
            model = vgg19(weights=VGG19_Weights.DEFAULT)
        elif learningtype=='normal':
            model = vgg19(weights=None)
        return model
    elif modelname=='vgg19_bn':
        from torchvision.models import vgg19_bn, VGG19_BN_Weights
        if learningtype=='transfer_learning':
            model = vgg19_bn(weights=VGG19_BN_Weights.DEFAULT)
            for param in model.parameters():
                param.requires_grad = False
        elif learningtype=='fine_tuning':
            model = vgg19_bn(weights=VGG19_BN_Weights.DEFAULT)
        elif learningtype=='normal':
            model = vgg19_bn(weights=None)
        return model


def modellist(): return model_list
def typelist(): return type_list
