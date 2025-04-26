import torch

'''def make_optimizer(args, model):
    params = []
    keys = []
    for key, value in model.named_parameters():
        if "Encode_text_img" in key:
            value.requires_grad_(False)
            continue
        if not value.requires_grad:
            continue
        lr = args.lr
        weight_decay = 1e-4
        if "bias" in key:
            lr = args.lr * 2
            weight_decay = 1e-4

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
        keys += [key]
    if args.optimizer_name == 'SGD':
        optimizer = getattr(torch.optim, args.optimizer_name)(params, momentum=args.momentum)
    else:
        optimizer = getattr(torch.optim, args.optimizer_name)(params)
    return optimizer'''


def make_optimizer(args, model):
    params = []
    keys = []
    for key, value in model.named_parameters():
        #print("key", key)
        '''if "text_encoder" in key:
            value.requires_grad_(False)
            continue
        if "image_encoder" in key:
            value.requires_grad_(False)
            continue'''
        if not value.requires_grad:
            continue
        lr = args.lr
        weight_decay = 1e-40
        if "bias" in key:
            lr = 2 * args.lr
            weight_decay = 1e-4

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
        keys += [key]

    if args.optimizer_name == 'SGD':
        optimizer = getattr(torch.optim, args.optimizer_name)(params, momentum=args.momentum)
    else:
        optimizer = getattr(torch.optim, args.optimizer_name)(params)
    return optimizer