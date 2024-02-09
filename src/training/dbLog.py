'''
    15 JAN, 2024
    Setting DB to watch logs(loss)
'''
import wandb

def set_db(args):
    if args.use_contrastive_loss:
        loss_group = 'contrastive loss'
    elif args.use_inter_loss:
        loss_group = 'inter loss'
    else:
        loss_group = 'basic loss'

    wandb.init(
        # name of wandb project
        project='clip-ssl',
        group=loss_group,
        name=args.name,
        notes=f'batch-{args.batch_size}_epoch-{args.epochs}_lr-{args.lr}_datasetType-{args.dataset_type}',
        tags=['clipself', loss_group, f'{args.batch_size}', f'{args.lr}', args.dataset_type],
        entity='sjiwoo'
    )

    wandb.config = {
        'learning_rate': args.lr,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'alpha': args.alpha
    }

    # wandb.define_metric('train_loss', summary='min')