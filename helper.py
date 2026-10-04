import logging
import sys # used for sys.stdout , tells the logger to write its message
# redirects it to same pipe as print 
import torch
import src.models.vision_transformer as vision_transformer
from src.utils.schedular import ( WarmupCosineSchedule, CosineWDSchedule)
from src.utils.tensors import trunc_normal_

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger() # everything below is written here

def load_checkpoint(
        device,
        r_path, # path to checkpoint file
        encoder,
        predictor,
        target_encoder,
        opt,
        scaler,
):
    try:
        checkpoint = torch.load(r_path, map_location=torch.device('cpu'))
        epoch = checkpoint['epoch']

        # -- loading encoder
        pretrained_dict = checkpoint['encoder']
        msg = encoder.load_state_dict(pretrained_dict)
        logger.info(f'loaded pretrained encoder from epoch {epoch} with msg: {msg}')

        # -- loading predictor
        pretrained_dict = checkpoint['predictor']
        msg = predictor.load_state_dict(pretrained_dict)
        logger.info(f'loaded pretrained predictor from epoch {epoch} with msg: {msg}')

        # -- loading target_encoder
        if target_encoder is not None:  # None when only evaluating
            pretrained_dict = checkpoint['target_encoder']
            msg = target_encoder.load_state_dict(pretrained_dict)
            logger.info(f'loaded pretrained target_encoder from epoch {epoch} with msg: {msg}')
        opt.load_state_dict(checkpoint['opt'])  # restores AdamW momentum/variance, not just weights
        if scaler is not None:
            scaler.load_state_dict(checkpoint['scaler'])
        logger.info(f'loaded optimizers from epoch {epoch}')
        logger.info(f'read-path: {r_path}')
        del checkpoint
    except Exception as e:
        # any failure (bad path, shape mismatch) silently restarts from epoch 0
        logger.info(f'Encountered exception when loading checkpoint {e}')
        epoch = 0
    return encoder, predictor, target_encoder, opt, scaler, epoch
 
 
def init_model(device, patch_size=16, model_name='vit_base', crop_size=224, pred_depth=6, pred_emb_dim=384):
    encoder = vision_transformer.__dict__[model_name](img_size=[crop_size], patch_size=patch_size)
    predictor = vision_transformer.__dict__['vit_predictor'](
        num_patches=encoder.patch_embed.num_patches,
        embed_dim=encoder.embed_dim,          # in/out width matches the encoder
        predictor_embed_dim=pred_emb_dim,     # narrower inside, so it's cheaper
        depth=pred_depth,
        num_heads=encoder.num_heads)
 
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=0.02)  # small random weights, extreme tails clipped
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):  # starts as pure normalisation (gamma=1, beta=0)
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)
 
    # NOTE: this overwrites any init already done inside vision_transformer.py
    for m in encoder.modules():
        init_weights(m)
    for m in predictor.modules():
        init_weights(m)
    encoder.to(device)
    predictor.to(device)
    logger.info(encoder)
    return encoder, predictor  # target encoder is made in train.py via copy.deepcopy(encoder)
 
 
def init_opt(encoder, predictor, iterations_per_epoch, start_lr, ref_lr, warmup, num_epochs,
             wd=1e-6, final_wd=1e-6, final_lr=0.0, use_bfloat16=False, ipe_scale=1.25):
    # Weight decay only on 2-D+ weight matrices; biases and 1-D params (LayerNorm) are excluded
    param_groups = [
        {'params': (p for n, p in encoder.named_parameters() if ('bias' not in n) and (len(p.shape) != 1))},
        {'params': (p for n, p in predictor.named_parameters() if ('bias' not in n) and (len(p.shape) != 1))},
        {'params': (p for n, p in encoder.named_parameters() if ('bias' in n) or (len(p.shape) == 1)),
         'WD_exclude': True, 'weight_decay': 0},  # WD_exclude: CosineWDSchedule skips this group
        {'params': (p for n, p in predictor.named_parameters() if ('bias' in n) or (len(p.shape) == 1)),
         'WD_exclude': True, 'weight_decay': 0},
    ]
    logger.info('Using AdamW')
    optimizer = torch.optim.AdamW(param_groups)  # lr/wd are set by the schedulers on their first step()
    # Steps are counted in iterations. ipe_scale=1.25 makes the schedule longer than the run,
    # so the LR never fully reaches final_lr
    total_steps = int(ipe_scale * num_epochs * iterations_per_epoch)
    scheduler = WarmupCosineSchedule(
        optimizer, warmup_steps=int(warmup * iterations_per_epoch),
        start_lr=start_lr, ref_lr=ref_lr, final_lr=final_lr, T_max=total_steps)
    wd_scheduler = CosineWDSchedule(optimizer, ref_wd=wd, final_wd=final_wd, T_max=total_steps)
    scaler = torch.cuda.amp.GradScaler() if use_bfloat16 else None
    return optimizer, scaler, scheduler, wd_scheduler