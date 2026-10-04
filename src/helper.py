import logging
import sys  # lets the logger print to the screen, same place as print() does
import torch
import src.models.vision_transformer as vision_transformer
from src.utils.schedular import ( WarmupCosineSchedule, CosineWDSchedule)
from src.utils.tensors import trunc_normal_

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()  # all the messages in this file go through this logger

# Loads a saved training run so we can continue from where we stopped
def load_checkpoint(
        device,
        r_path,  # where the saved file is
        encoder,
        predictor,
        target_encoder,
        opt,
        scaler,
):
    try:
        checkpoint = torch.load(r_path, map_location=torch.device('cpu'))
        epoch = checkpoint['epoch']  # which epoch the save was made at

        # load the encoder's saved weights
        pretrained_dict = checkpoint['encoder']
        msg = encoder.load_state_dict(pretrained_dict)
        logger.info(f'loaded pretrained encoder from epoch {epoch} with msg: {msg}')

        # load the predictor's saved weights
        pretrained_dict = checkpoint['predictor']
        msg = predictor.load_state_dict(pretrained_dict)
        logger.info(f'loaded pretrained predictor from epoch {epoch} with msg: {msg}')

        # load the target encoder too (we skip it when we're only testing, not training)
        if target_encoder is not None:
            pretrained_dict = checkpoint['target_encoder']
            msg = target_encoder.load_state_dict(pretrained_dict)
            logger.info(f'loaded pretrained target_encoder from epoch {epoch} with msg: {msg}')
        # the optimizer also has its own memory of past steps, so we load that back too
        opt.load_state_dict(checkpoint['opt'])
        if scaler is not None:
            scaler.load_state_dict(checkpoint['scaler'])
        logger.info(f'loaded optimizers from epoch {epoch}')
        logger.info(f'read-path: {r_path}')
        del checkpoint  # free up memory
    except Exception as e:
        # if anything goes wrong (wrong path, sizes don't match), we just start fresh from epoch 0
        logger.info(f'Encountered exception when loading checkpoint {e}')
        epoch = 0
    return encoder, predictor, target_encoder, opt, scaler, epoch


# Builds the encoder and predictor models
def init_model(device, patch_size=16, model_name='vit_base', crop_size=224, pred_depth=6, pred_emb_dim=384):
    # pick the model by its name, e.g. 'vit_base' calls vit_base()
    encoder = vision_transformer.__dict__[model_name](img_size=[crop_size], patch_size=patch_size)
    predictor = vision_transformer.__dict__['vit_predictor'](
        num_patches=encoder.patch_embed.num_patches,
        embed_dim=encoder.embed_dim,          # takes in and gives out the same size as the encoder
        predictor_embed_dim=pred_emb_dim,     # smaller on the inside, so it runs faster
        depth=pred_depth,
        num_heads=encoder.num_heads)

    # sets the starting values of the weights before training
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=0.02)  # small random numbers, no very big ones
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):  # start LayerNorm so it doesn't change anything yet
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    # note: this replaces whatever starting weights vision_transformer.py already set
    for m in encoder.modules():
        init_weights(m)
    for m in predictor.modules():
        init_weights(m)
    encoder.to(device)  # move the models to the GPU or CPU
    predictor.to(device)
    logger.info(encoder)
    return encoder, predictor  # the target encoder is made later in train.py as a copy of the encoder


# Sets up the optimizer and the schedules that change the learning rate and weight decay over time
def init_opt(encoder, predictor, iterations_per_epoch, start_lr, ref_lr, warmup, num_epochs,
             wd=1e-6, final_wd=1e-6, final_lr=0.0, use_bfloat16=False, ipe_scale=1.25):
    # Split the weights into two groups:
    #   - normal weight matrices get weight decay (a small pull towards zero to avoid overfitting)
    #   - biases and LayerNorm values don't get weight decay
    param_groups = [
        {'params': (p for n, p in encoder.named_parameters() if ('bias' not in n) and (len(p.shape) != 1))},
        {'params': (p for n, p in predictor.named_parameters() if ('bias' not in n) and (len(p.shape) != 1))},
        {'params': (p for n, p in encoder.named_parameters() if ('bias' in n) or (len(p.shape) == 1)),
         'WD_exclude': True, 'weight_decay': 0},  # WD_exclude tells the weight decay schedule to leave this group alone
        {'params': (p for n, p in predictor.named_parameters() if ('bias' in n) or (len(p.shape) == 1)),
         'WD_exclude': True, 'weight_decay': 0},
    ]
    logger.info('Using AdamW')
    optimizer = torch.optim.AdamW(param_groups)  # the schedules below set the actual learning rate and weight decay
    # Total number of training steps. ipe_scale=1.25 makes the schedule a bit longer than training,
    # so the learning rate never quite drops all the way to final_lr
    total_steps = int(ipe_scale * num_epochs * iterations_per_epoch)
    # learning rate: slowly goes up during warmup, then slowly comes down in a smooth curve
    scheduler = WarmupCosineSchedule(
        optimizer, warmup_steps=int(warmup * iterations_per_epoch),
        start_lr=start_lr, ref_lr=ref_lr, final_lr=final_lr, T_max=total_steps)
    # weight decay: changes smoothly from wd to final_wd over training
    wd_scheduler = CosineWDSchedule(optimizer, ref_wd=wd, final_wd=final_wd, T_max=total_steps)
    # the scaler is only needed when training with lower-precision numbers (bfloat16)
    scaler = torch.cuda.amp.GradScaler() if use_bfloat16 else None
    return optimizer, scaler, scheduler, wd_scheduler
