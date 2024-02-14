import logging
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as Ft
from torchvision.transforms import InterpolationMode

from training.dist_utils import all_gather
from tqdm import tqdm
from .distributed import is_master
from open_clip import get_cast_dtype
from .precision import get_autocast

def IoU(preds, gt_masks, results:dict)->dict:
        min_stuff_idx = 80

        for pred, gt_mask in zip(preds, gt_masks):
            anns = gt_mask.unique()
            anns_is_thing = anns < min_stuff_idx
            
            for lb, is_thing in zip(anns, anns_is_thing):               
                lb_pred = (pred == lb).sum(-1, keepdim=True)
                lb_gt_mask = (gt_mask == lb)
                
                inter = (lb_pred * lb_gt_mask)
                union = (lb_pred + lb_gt_mask) * 1 - (inter * 1)
                iou = (inter.sum() / union.sum())
                
                category_1 = 'thing' if is_thing else 'stuff'
                if lb in results.keys():
                    results[category_1][lb.item()].append(iou.item())
                else:
                    results[category_1][lb.item()] = [iou.item()]

def mIoU_with_is_thing_and_cls(correct_matrix, gt_masks, top1_results, top5_results):
    IoU(correct_matrix[:, :, :, :1], gt_masks, top1_results)
    IoU(correct_matrix, gt_masks, top5_results)

def run(model, dataloader, args, is_miou):
    patch_size = dataloader.dataset.downsample_factor
    cls_embeddings = dataloader.dataset.embeddings
    cls_embeddings = F.normalize(torch.from_numpy(cls_embeddings).float(), dim=-1)
    cls_embeddings = cls_embeddings.to(args.device)
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)
    if cast_dtype is not None:
        cls_embeddings = cls_embeddings.to(dtype=cast_dtype)
    with torch.no_grad():
        correct_rois = []
        correct_maskpool = []
        correct_crops = []
        # similarity_crops = []
        # similarity_rois = []
        # similarity_maskpool = []
        # all_box_sizes = []
        all_is_thing = []
        all_cls_labels = []
        top1_iou_results = {
            'thing': dict(),
            'stuff': dict()
        }
        top5_iou_results = {
            'thing': dict(),
            'stuff': dict()
        }
        
        for images, bboxes, image_crops, gt_masks, gt_seg_map, _ \
                in tqdm(dataloader, disable=not is_master(args)):
            images = images.to(args.device)
            bboxes = bboxes.to(args.device)
            image_crops = image_crops.to(args.device)
            # masked_image_crops = masked_image_crops.to(args.device)
            gt_masks = gt_masks.to(args.device)
            gt_seg_map = gt_seg_map.to(args.device)
            if cast_dtype is not None:
                images = images.to(dtype=cast_dtype)
                bboxes = bboxes.to(dtype=cast_dtype)
                image_crops = image_crops.to(dtype=cast_dtype)
                # masked_image_crops = masked_image_crops.to(dtype=cast_dtype)
                gt_masks = gt_masks.to(dtype=cast_dtype)
                gt_seg_map = gt_seg_map.to(dtype=cast_dtype)
            image_crops_list = []
            gt_masks_list = []
            cls_labels = []
            rois = []
            box_sizes = []
            is_thing = []
            # for each data in a batch
            # for bboxes_per_image, crops_per_image, gt_mask, masked_crops_per_image \
            #         in zip(bboxes, image_crops, gt_masks, masked_image_crops):
            for bboxes_per_image, crops_per_image, gt_mask \
                    in zip(bboxes, image_crops, gt_masks):
                valid = bboxes_per_image[:, 5] > 0.5
                rois.append(bboxes_per_image[valid, :4])
                cls_labels.append(bboxes_per_image[valid, 4])
                image_crops_list.append(crops_per_image[valid])
                gt_masks_list.append(gt_mask[valid])
                box_sizes.append(bboxes_per_image[valid, 6])
                is_thing.append(bboxes_per_image[valid, 7])
            cls_labels = torch.cat(cls_labels, dim=0).to(torch.long)
            if cls_labels.shape[0] == 0:
                continue
            image_crops = torch.cat(image_crops_list)
            box_sizes = torch.cat(box_sizes, dim=0).float()
            is_thing = torch.cat(is_thing, dim=0)
            # all_box_sizes.append(box_sizes)
            all_is_thing.append(is_thing)
            with autocast():
                # predict
                if args.distributed and not args.horovod:
                    module = model.module
                else:
                    module = model
                roi_extractor = module.encode_pseudo_boxes
                roi_features = roi_extractor(images, rois, normalize=True,
                                             extract_type=args.extract_type)
                mask_pooler = module.encode_masks
                maskpool_features = mask_pooler(images, gt_masks_list,
                                                normalize=True, mask_attn=args.extract_type == "v1")
                
                # New way to obtain crop features
                if args.image_ave_pool:
                    feature_map = module.visual.encode_dense(image_crops, keep_shape=True)
                    crop_features = feature_map.mean(dim=(-2, -1))
                    crop_features = F.normalize(crop_features, dim=-1)
                else:
                    crop_features = module.encode_image(image_crops, normalize=True)

                if cast_dtype is not None:
                    roi_features = roi_features.to(dtype=cast_dtype)
                    crop_features = crop_features.to(dtype=cast_dtype)
                    maskpool_features = maskpool_features.to(dtype=cast_dtype)

                roi_logits = roi_features @ cls_embeddings.T
                crop_logits = crop_features @ cls_embeddings.T
                maskpool_logits = maskpool_features @ cls_embeddings.T

            _, roi_top5_inds = roi_logits.topk(5)
            _, crop_top5_inds = crop_logits.topk(5)
            _, maskpool_top5_inds = maskpool_logits.topk(5)
            correct_rois.append(roi_top5_inds == cls_labels.view(-1, 1))
            correct_crops.append(crop_top5_inds == cls_labels.view(-1, 1))
            correct_maskpool.append(maskpool_top5_inds == cls_labels.view(-1, 1))

            # similarity_rois.append(torch.gather(roi_logits, dim=1, index=cls_labels.view(-1, 1))[:, 0])
            # similarity_crops.append(torch.gather(crop_logits, dim=1, index=cls_labels.view(-1, 1))[:, 0])
            # similarity_maskpool.append(torch.gather(maskpool_logits, dim=1, index=cls_labels.view(-1, 1))[:, 0])

            all_cls_labels.append(cls_labels)
            
            '''
                modify
                batch size must be 1
            '''
            if is_miou:
                with autocast():
                    feature_map = module.visual.encode_dense(image_crops, keep_shape=True)
                    if cast_dtype is not None:
                        feature_map = feature_map.to(dtype=cast_dtype)
                    dense_logits = torch.einsum("bdhw,cd->bchw", feature_map, cls_embeddings)

                _, h, w = gt_seg_map.shape # [bs, h, w]
                gt_seg_map = gt_seg_map.unsqueeze(-1)

                resize_dense_logits = Ft.resize(dense_logits, (h, w), interpolation=InterpolationMode.BILINEAR) # [bs, class, h, w]
                resize_dense_logits = F.softmax(resize_dense_logits, dim=1)
                resize_dense_logits = resize_dense_logits.permute(0,2,3,1) # [bs, h, w, class]

                _, dense_top5_inds = resize_dense_logits.topk(5)   

                mIoU_with_is_thing_and_cls(dense_top5_inds, gt_seg_map, top1_iou_results, top5_iou_results)

        # TODO: gather correct matrix across gpus
        correct_rois = torch.cat(correct_rois).float()
        correct_crops = torch.cat(correct_crops).float()
        correct_maskpool = torch.cat(correct_maskpool).float()
        # similarity_rois = torch.cat(similarity_rois).float()
        # similarity_crops = torch.cat(similarity_crops).float()
        # similarity_maskpool = torch.cat(similarity_maskpool).float()
        # all_box_sizes = torch.cat(all_box_sizes)
        all_is_thing = torch.cat(all_is_thing)
        all_cls_labels = torch.cat(all_cls_labels)
        
        # gt_masks_list = torch.cat(gt_masks_list)
        if args.distributed and not args.horovod:
            correct_rois = multi_gpu_sync(correct_rois)
            correct_crops = multi_gpu_sync(correct_crops)
            correct_maskpool = multi_gpu_sync(correct_maskpool)
            # all_box_sizes = multi_gpu_sync(all_box_sizes)
            all_is_thing = multi_gpu_sync(all_is_thing)
            # similarity_rois = multi_gpu_sync(similarity_rois)
            # similarity_crops = multi_gpu_sync(similarity_crops)
            # similarity_maskpool = multi_gpu_sync(similarity_maskpool)
            all_cls_labels = multi_gpu_sync(all_cls_labels)

    # return correct_rois, correct_crops, correct_maskpool, \
    #     similarity_rois, similarity_crops, similarity_maskpool, \
    #     all_box_sizes, all_is_thing, all_cls_labels, (top1_iou_results, top5_iou_results)
        return correct_rois, correct_crops, correct_maskpool, all_is_thing, all_cls_labels, (top1_iou_results, top5_iou_results)

def multi_gpu_sync(x):
    device = x.device
    x_list = all_gather(x.cpu())
    x = torch.cat([res.to(device) for res in x_list])
    return x


def macc_with_is_thing(correct_matrix, is_thing, all_cls_labels, prefix):
    def _macc(corrects, cls_labels):
        min_id = cls_labels.min().item()
        max_id = cls_labels.max().item()
        cand_labels = list(range(min_id, max_id+1))

        acc_per_cls = []

        for lb in cand_labels:
            corrects_per_cls = corrects[cls_labels == lb]
            if corrects_per_cls.shape[0] == 0:
                continue
            acc_per_cls.append(corrects_per_cls.mean().half().item())

        return sum(acc_per_cls) / len(acc_per_cls)

    results = {}
    thing_correct_matrix = correct_matrix[is_thing > 0]
    stuff_correct_matrix = correct_matrix[is_thing < 1]

    thing_cls_labels = all_cls_labels[is_thing > 0].long()
    stuff_cls_labels = all_cls_labels[is_thing < 1].long()

    thing_top1_acc = _macc(thing_correct_matrix[:, 0], thing_cls_labels)
    thing_top5_acc = _macc(thing_correct_matrix.sum(-1), thing_cls_labels)

    stuff_top1_acc = _macc(stuff_correct_matrix[:, 0], stuff_cls_labels)
    stuff_top5_acc = _macc(stuff_correct_matrix.sum(-1), stuff_cls_labels)

    results[f'{prefix}.thing.macc1'] = thing_top1_acc
    results[f'{prefix}.thing.macc5'] = thing_top5_acc
    results[f'{prefix}.stuff.macc1'] = stuff_top1_acc
    results[f'{prefix}.stuff.macc5'] = stuff_top5_acc

    return results


def zero_shot_eval(model, data, epoch, args):
    def _cal_mIoU(results, topk):
        thing_results, stuff_results = results['thing'], results['stuff']
        thing_iou, stuff_iou = 0., 0.
        num_thing, num_stuff = 0, 0
        for ann, ious in thing_results.items():
            miou = sum(ious) / len(ious)
            thing_results.update({ann: miou})
            thing_iou += miou
            num_thing += 1
        for ann, ious in stuff_results.items():
            miou = sum(ious) / len(ious)
            stuff_results.update({ann: miou})
            stuff_iou += miou
            num_stuff += 1
            
        thing_miou = thing_iou / num_thing if num_thing else -1
        stuff_miou = stuff_iou / num_stuff if num_stuff else -1
        
        return {f'thing.mIoU.{topk}': thing_miou,
                f'stuff.mIoU.{topk}': stuff_miou}
        
    if 'val' not in data:
        logging.info('no val data')
        return {}
    if args.zeroshot_frequency == 0:
        logging.info('zeroshot_freq == 0')
        return {}
    if (epoch % args.zeroshot_frequency) != 0 and epoch != args.epochs:
        logging.info(f'{epoch} % {args.zeroshot_frequency} != 0 && {epoch} != {args.epochs}')
        return {}
    is_miou = False if epoch % args.zeroshot_frequency else True
    logging.info('Region classifier')
    results = {}
    # correct_rois, correct_crops, correct_maskpool, \
    #     similarity_rois, similarity_crops, similarity_maskpool, \
    #     all_box_sizes, all_is_thing, all_cls_labels, maskpool_result = run(model, data['val'].dataloader, args)
    correct_rois, correct_crops, correct_maskpool, all_is_thing, all_cls_labels, iou_result = run(model, data['val'].dataloader, args, is_miou)
    
    results.update(macc_with_is_thing(correct_rois, all_is_thing, all_cls_labels, 'rois'))
    results.update(macc_with_is_thing(correct_crops, all_is_thing, all_cls_labels, 'crops'))
    results.update(macc_with_is_thing(correct_maskpool, all_is_thing, all_cls_labels, 'maskpool'))
    if is_miou:
        results.update(_cal_mIoU(iou_result[0], 'top1'))
        results.update(_cal_mIoU(iou_result[1], 'top5'))

    return results
