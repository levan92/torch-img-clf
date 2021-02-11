from torchvision import transforms

def build_transforms(aug_dicts):

    composing_augs = []
    for aug_dict in aug_dicts:
        augs = list(aug_dict.items())
        assert len(augs) == 1,'Invalid format in aug yaml file'
        aug_name, aug_kwargs = augs[0]
        # print(aug_name, aug_kwargs)
        wanted_aug_fn = getattr(transforms, aug_name)
        if aug_kwargs:
            wanted_aug = wanted_aug_fn(**aug_kwargs)
        else:
            wanted_aug = wanted_aug_fn()
        composing_augs.append(wanted_aug)

    composed_transforms = transforms.Compose(composing_augs) 
    print(composed_transforms)
    return composed_transforms