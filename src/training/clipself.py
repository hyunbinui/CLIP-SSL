import random
import torch
import torch.nn.functional as F
import logging

import wandb

class CLIPSelf:
    def __init__(self, args=None):
        self.iter = 0
        self.epoch = 0
        self.num_samples = 0
        self.num_pos_samples = 0
        if args.use_contrastive_loss:
            self.intra_grid_patch = 'intra_grid_patch' in args.neg_type
            self.inter_grid = 'inter_grid_patch' in args.neg_type or 'inter_grid_cls' in args.neg_type
            self.inter_grid_patch = 'inter_grid_patch' in args.neg_type
            self.inter_grid_cls = 'inter_grid_cls' in args.neg_type
            self.intra_batch = 'intra_batch_patch' in args.neg_type or 'intra_batch_cls' in args.neg_type
            self.intra_batch_patch = 'intra_batch_patch' in args.neg_type
            self.intra_batch_cls = 'intra_batch_cls' in args.neg_type

    def print_num_samples(self):
        print('------------------------------------------------------------------------------------------------------------------------')
        print(f'num_samples: {self.num_samples},    num_neg_samples: {[self.num_samples - num_pos for num_pos in self.num_pos_samples]}')
        print(f'neg type: inter_grid {self.inter_grid}, intra_grid_patch {self.inter_grid_patch}, inter_grid_patch {self.inter_grid_patch}, inter_grid_cls {self.inter_grid_cls}')
        print('------------------------------------------------------------------------------------------------------------------------')

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

        if args.use_contrastive_loss:
            '''
                2 JAN 24: add contrastive loss
            '''
            image_crops = torch.cat(crops_list)
            
            with torch.no_grad():
                teacher_crop_features = dist_model.encode_image(image_crops, normalize=False)
            student_dense_features = model.encode_dense(images, normalize=False, keep_shape=True)

            normed_student_features = F.normalize(student_dense_features, dim=1)
            normed_teacher_features = F.normalize(teacher_crop_features, dim=-1)

            denormed_boxes = self._denormalize_boxes(rois_list, student_dense_features)

            # contrastive_loss = self.infonce(args, normed_student_features, normed_teacher_features, denormed_boxes, temperature=model.logit_scale.exp())
            contrastive_loss = self.infonce(args, normed_student_features, normed_teacher_features, denormed_boxes)
            losses = dict(contrastive_loss=contrastive_loss)

        elif args.use_inter_loss:
            image_crops = torch.cat(crops_list)
            
            with torch.no_grad():
                teacher_crop_features = dist_model.encode_image(image_crops, normalize=False)
            # basic loss
            student_roi_features = model.encode_pseudo_boxes(images, rois_list, normalize=False,
                                                            extract_type=args.extract_type)

            normed_student_features = F.normalize(student_roi_features, dim=-1)
            normed_teacher_features = F.normalize(teacher_crop_features, dim=-1)

            loss_cosine = 1.0 - (normed_student_features *
                                normed_teacher_features).sum(-1).mean()

            # auxilarary loss (inter loss)
            student_dense_features = model.encode_dense(images, normalize=False, keep_shape=True)
            normed_student_features = F.normalize(student_dense_features, dim=-1)

            denormed_boxes = self._denormalize_boxes(rois_list, student_dense_features)

            loss_inter = self.loss_inter_features(denormed_boxes, student_dense_features)

            losses = dict(loss_cosine=loss_cosine*args.cosine_weight, loss_inter=loss_inter * 0.1)
        else:
            '''
                basic clipself loss
            '''
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
    
    def infonce(self, args, dense_features, crop_features, denormed_boxes, temperature=1., reduction='mean'):
        crop_idx = 0
        loss = []

        # if 'intra_batch_patch' in args.neg_type:
        #     intra_batch_patch_dense_features = []
        # if 'intra_batch_cls' in args.neg_type:
        #     intra_batch_grid_cls_feature = []

        # for all images in a batch
        for i in range(len(denormed_boxes)):
            boxes_single_image = denormed_boxes[i].round().int()

            if self.inter_grid:
                inter_grid_patch_dense_features = []
                num_patch_per_image_grid = []
                inter_grid_cls_features = []
                pos_sims = []
   
            # for all boxes in an image
            for j in range(len(boxes_single_image)):
                box = boxes_single_image[j]

                # print(f'box values: {box}, crop_feature: {crop_features.shape}')

                crop_feature = crop_features[crop_idx][None, ...]
                
                box_dense_features = dense_features[i, :, box[0]:box[2], box[1]:box[3]]
                # shape of box_dense_features: [num of patches per box, dim]
                box_dense_features = box_dense_features.reshape(box_dense_features.shape[0], -1).transpose(0, 1)

                # shape of pos_sim: [num of patches per box, 1]
                pos_sim = torch.matmul(box_dense_features, crop_feature.T)
                pos_sims.append(pos_sim)
                
                if self.inter_grid:
                    num_patch_per_image_grid.append(box_dense_features.shape[0])
                    inter_grid_patch_dense_features.append(box_dense_features)
                    if self.inter_grid_cls:
                        inter_grid_cls_features.append(crop_feature)
                elif self.intra_grid_patch:
                    # shape of sim_matrix: [num of patches per box, num of patches per box + 1]
                    neg_sim = torch.matmul(box_dense_features, box_dense_features.T).fill_diagonal_(0)
                    sim_matrix = torch.cat([pos_sim, neg_sim], axis=1)

                    '''
                        21 JAN, 2024
                        use subset of negative sample sorted by similarity with target patch embedding
                    '''
                    if args.start_neg_ratio != 1.:
                        assert 0 < args.start_neg_ratio < 1, f'start_neg_ratio is not a value between 0 and 1: {args.start_neg_ratio}'
                        sorted_sim_matrix, _ = torch.sort(sim_matrix[:,1:], dim=-1)
                        # start from 0.5 and increase by 0.1 at every epoch to 1.
                        neg_ratio = min(1, (args.start_neg_ratio + self.epoch * 0.1))
                        neg_samples = int(sorted_sim_matrix.shape[-1] * neg_ratio)
                        sim_matrix = torch.concat([sim_matrix[:,0:1], sorted_sim_matrix[:,:neg_samples]], axis=-1)

                    label = torch.zeros(sim_matrix.shape[0], dtype=torch.long, device=sim_matrix.device)

                    loss.append(F.cross_entropy(sim_matrix*temperature, label, reduction=reduction))
                
                crop_idx += 1

            if self.inter_grid:
                inter_grid_patch_dense_features = torch.cat(inter_grid_patch_dense_features, axis=0)
                # print(f'all patch dense feature shape: {inter_grid_patch_dense_features.shape}')

                sim_matrix = torch.cat(pos_sims, axis=0)
                if self.inter_grid_patch:
                    neg_sim = torch.matmul(inter_grid_patch_dense_features, inter_grid_patch_dense_features.T)
                    start, end = 0, 0
                    for num_patch_per_grid in num_patch_per_image_grid:
                        end += num_patch_per_grid
                        # set similarity within same grid to 0 (features in same grid are not negative sample each other)
                        neg_sim[start:end, start:end] = float('-inf')
                        start = end
                    sim_matrix = torch.cat([sim_matrix, neg_sim], axis=1)
                if self.inter_grid_cls:
                    inter_grid_cls_features = torch.cat(inter_grid_cls_features, axis=0)
                    neg_sim = torch.matmul(inter_grid_patch_dense_features, inter_grid_cls_features.T)
                    start, end = 0, 0
                    # print(f'num_patch_per_image_grid: {len(num_patch_per_image_grid)}, inter_grid_cls_features: {inter_grid_cls_features.shape}, neg_sim shape: {neg_sim.shape}')
                    for grid_idx, num_patch_per_grid in enumerate(num_patch_per_image_grid):
                        end += num_patch_per_grid
                        # set similarity with its own grid to 0 (positive similarity is calculated separately)
                        neg_sim[start:end, grid_idx] = float('-inf')
                        start = end
                    sim_matrix = torch.cat([sim_matrix, neg_sim], axis=1)

                self.num_samples = sim_matrix.shape[0]
                self.num_pos_samples = num_patch_per_image_grid

                label = torch.zeros(sim_matrix.shape[0], dtype=torch.long, device=sim_matrix.device)

                loss.append(F.cross_entropy(sim_matrix*temperature, label, reduction=reduction))

            if self.iter % 5000:
                target_patch = num_patch_per_image_grid[0]
                sim_dic = {
                        "positive sim": sim_matrix[target_patch][0],
                        "hard neg sim": sim_matrix[target_patch][target_patch+1],
                        "easy neg sim": sim_matrix[target_patch][-1]
                    }
                wandb.log(sim_dic, step=self.iter)
                self.iter += 1

        return sum(loss) / len(loss)
    
    '''
        8 JAN 24: add inter loss
    '''    
    def loss_inter_features(self, denormed_boxes, dense_features):
        result = 0
        count = 0
        
        for i in range(len(denormed_boxes)):
            boxes_single_image = denormed_boxes[i].round().int()
            for j in range(len(boxes_single_image)):
                box = boxes_single_image[j]
                
                cropped_features = dense_features[i, :, box[0]:box[2], box[1]:box[3]]
                #avg_features = cropped_features.reshape(cropped_features.shape[0], -1).mean(1)
                cropped_features = cropped_features.reshape(cropped_features.shape[0], -1).transpose(0, 1)
                dot_matrix = torch.matmul(cropped_features, cropped_features.T).fill_diagonal_(0) #.sum() / 2
                result += torch.mean(dot_matrix)
                count += 1

        return 1-(result / count)
