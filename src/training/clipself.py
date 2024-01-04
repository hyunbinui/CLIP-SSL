import random
import torch
import torch.nn.functional as F

class CLIPSelf:
    def __call__(self, batch, model, dist_model, loss, device, cast_dtype, distributed, args):
        if distributed:
            model = model.module
            dist_model = dist_model.module
        images, normed_boxes, image_crops = batch       # note texts are not paired with images

        images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
        normed_boxes = normed_boxes.to(device=device, dtype=cast_dtype, non_blocking=True)
        image_crops = image_crops.to(device=device, dtype=cast_dtype, non_blocking=True)

        if args.multiscale:
            cur_h, cur_w = images.shape[2:]
            assert cur_h == cur_w
            if cur_h == 1024:
                tar_sizes = [320, 640, 896, 1024]
            elif cur_h == 896:
                tar_sizes = [336, 448, 672, 896]
            else:
                raise NotImplementedError
            tar_size = random.choice(tar_sizes)
            images = F.interpolate(images, size=(tar_size, tar_size), mode='bilinear')

        rois_list = []
        crops_list = []
        for bboxes_per_image, crops_per_image in zip(normed_boxes, image_crops):
            valid = bboxes_per_image[:, -1] > 0.5
            rois_list.append(bboxes_per_image[valid, :4])
            crops_list.append(crops_per_image[valid])

        if args.use_contrastive_loss is False:
            image_crops = torch.cat(crops_list)

            with torch.no_grad():
                teacher_crop_features = dist_model.encode_image(image_crops, normalize=False)
            student_roi_features = model.encode_pseudo_boxes(images, rois_list, normalize=False,
                                                            extract_type=args.extract_type)

            normed_student_features = F.normalize(student_roi_features, dim=-1)
            normed_teacher_features = F.normalize(teacher_crop_features, dim=-1)

            loss_cosine = 1.0 - (normed_student_features *
                                normed_teacher_features).sum(-1).mean()
            losses = dict(loss_cosine=loss_cosine*args.cosine_weight)
        else:
            '''
                2 JAN 24: add contrastive loss
            '''
            image_crops = torch.cat(crops_list)
            
            with torch.no_grad():
                teacher_crop_features = dist_model.encode_image(image_crops, normalize=False)
            student_dense_features = model.encode_dense(images, normalize=False, keep_shape=True)

            normed_student_features = F.normalize(student_dense_features, dim=-1)
            normed_teacher_features = F.normalize(teacher_crop_features, dim=-1)

            denormed_boxes = self._denormalize_boxes(rois_list, student_dense_features)

            contrastive_loss = self.infonce(normed_student_features, normed_teacher_features, denormed_boxes)

            losses = dict(contrastive_loss=contrastive_loss)

        return losses, len(images), model.logit_scale.exp()

    def _denormalize_boxes(self, normed_boxes, x):
        h, w = x.shape[-2:]
        denormed_boxes = []
        for boxes in normed_boxes:
            new_boxes = boxes.clone()   # FIXME: do not change the value in normed_boxes!
            new_boxes[:, [0, 2]] *= w
            new_boxes[:, [1, 3]] *= h
            denormed_boxes.append(new_boxes)
        return denormed_boxes
    
    def infonce(self, dense_features, crop_features, denormed_boxes, temperature=0.1, reduction='mean'):
        
        # logits, labels = [], []
        crop_idx = 0
        loss = []

        # for all images in a batch
        for i in range(len(denormed_boxes)):
            boxes_single_image = denormed_boxes[i].round().int()
   
            # for all boxes in an image
            for j in range(len(boxes_single_image)):
                box = boxes_single_image[j]
                crop_feature = crop_features[crop_idx][None, ...]
                crop_idx += 1
                
                box_dense_features = dense_features[i, :, box[0]:box[2], box[1]:box[3]]
                # shape of box_dense_features: [num of patches per box, dim]
                box_dense_features = box_dense_features.reshape(box_dense_features.shape[0], -1).transpose(0, 1)
                # concat CLS token of the box
                box_dense_features = torch.cat([crop_feature, box_dense_features], dim=0)
                
                # shape of sim_matrix: [num of patches per box, num of patches per box + 1]
                sim_matrix = torch.matmul(box_dense_features, box_dense_features.T).fill_diagonal_(0)[1:,:]
                label = torch.zeros(sim_matrix.shape[0], dtype=torch.long, device=sim_matrix.device)

                loss.append(F.cross_entropy(sim_matrix / temperature, label, reduction=reduction))

        return sum(loss) / len(loss)