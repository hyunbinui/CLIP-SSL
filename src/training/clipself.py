import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

import wandb

'''
    31 JAN 24: add deep infomax - localDIM
'''

class LocalDim(nn.Module):
    def __init__(self):
        super().__init__()
        self.c0 = nn.Conv2d(192, 512, kernel_size=1)
        self.c1 = nn.Conv2d(512, 512, kernel_size=1)
        self.c2 = nn.Conv2d(512, 1, kernel_size=1)

    def forward(self, x, device):
        # h = F.relu(self.c0(x))
        # h = F.relu(self.c1(h))
        # return self.c2(h)
        l1 = nn.Linear(x.shape[0],1).to(device)
        x = l1(x.T)
        return x.T

class CLIPSelf:
    # def __init__(self):
    #     super.__init__()
    #     self.local_dim = LocalDim()

    def __call__(self, batch, model, dist_model, loss, device, cast_dtype, distributed, args):
        if distributed:
            model = model.module
            dist_model = dist_model.module
        images, normed_boxes, image_crops = batch       # note texts are not paired with images
        
        images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
        normed_boxes = normed_boxes.to(device=device, dtype=cast_dtype, non_blocking=True)
        image_crops = image_crops.to(device=device, dtype=cast_dtype, non_blocking=True)
        #self.local_dim = self.local_dim.to(device=device)

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
        if args.multi_grid_train:
            for bboxes_per_image, crops_per_image in zip(normed_boxes, image_crops):
                temp_crops_list = []
                for i in range(image_crops.shape[1]):
                    valid = bboxes_per_image[i][:, -1] > 0.5
                    temp_crops_list.append(crops_per_image[i][valid])
                crops_list.append(temp_crops_list)
        else:
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
                
            contrastive_loss = self.infonce(normed_student_features, normed_teacher_features, denormed_boxes)
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

        elif args.use_local_infomax_loss:
            '''
                28 JAN 24: add deepInfomax_local infomax loss
            '''
            image_crops = torch.cat(crops_list)

            with torch.no_grad():
                teacher_crop_features = dist_model.encode_image(image_crops, normalize=False)
            student_dense_features = model.encode_dense(images, normalize=False, keep_shape=True)

            normed_student_features = F.normalize(student_dense_features, dim=-1)
            normed_teacher_features = F.normalize(teacher_crop_features, dim=-1)
            denormed_boxes = self._denormalize_boxes(rois_list, student_dense_features)
                
            contrastive_loss = self.local_infomax(model, normed_student_features, normed_teacher_features, denormed_boxes, device)
            losses = dict(contrastive_loss=contrastive_loss)

        elif args.multi_grid_train:
            '''
                7 FEB 24: add multi grid training method
            '''
            crops_list = [torch.cat(crops, dim=0) for crops in zip(*crops_list)]
            normed_cls_features = []
            
            with torch.no_grad():
                g1_cls_teacher = dist_model.encode_image(crops_list[0], normalize=False)
                g2_cls_teacher = dist_model.encode_image(crops_list[1], normalize=False)
            
            g2_cls_student = model.encode_image(crops_list[1], normalize=False)
            g3_cls_student = model.encode_image(crops_list[2], normalize=False)
            
            normed_g1_teacher = F.normalize(g1_cls_teacher, dim=-1)
            normed_g2_teacher = F.normalize(g2_cls_teacher, dim=-1)

            normed_g2_student = F.normalize(g2_cls_student, dim=-1)
            normed_g3_student = F.normalize(g3_cls_student, dim=-1)
            
            batch_size = args.batch_size
            g1_g2_loss_cosine = self.get_loss_cosine(normed_g1_teacher, normed_g2_student, batch_size)
            g1_g3_loss_cosine = self.get_loss_cosine(normed_g1_teacher, normed_g3_student, batch_size)
            g2_g3_loss_cosine = self.get_loss_cosine(normed_g2_teacher, normed_g3_student, batch_size)

            final_loss_cosine = (g1_g2_loss_cosine + g1_g3_loss_cosine + g2_g3_loss_cosine) / 3
            losses = dict(loss_cosine=final_loss_cosine*args.cosine_weight)

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

    def get_loss_cosine(self, grid_a, grid_b, batch_size=4):
        grid_a_num, grid_b_num = grid_a.shape[0] // batch_size, grid_b.shape[0] // batch_size
        grid_num_differ = grid_b_num // grid_a_num
        for idx in range(grid_a_num):
            grid_loss = 0
            for b in range(batch_size):
                repeated_grid_a = grid_a[idx + b*grid_a_num].repeat(grid_num_differ, 1)
                grid_b_start_idx = idx*grid_num_differ+b*grid_b_num
                temp_loss = 1.0 - (
                                repeated_grid_a * 
                                grid_b[grid_b_start_idx:grid_b_start_idx+grid_num_differ][:]
                            ).sum(-1).mean()
                grid_loss += temp_loss
            grid_loss /= batch_size
        return grid_loss / grid_num_differ

    def _denormalize_boxes(self, normed_boxes, x):
        h, w = x.shape[-2:]
        denormed_boxes = []
        for boxes in normed_boxes:
            new_boxes = boxes.clone()   # FIXME: do not change the value in normed_boxes!
            new_boxes[:, [0, 2]] *= w
            new_boxes[:, [1, 3]] *= h
            denormed_boxes.append(new_boxes)
        return denormed_boxes
    
    '''
        8 JAN 24: add inter loss
    '''  
    def local_infomax(self, model, dense_features, crop_features, denormed_boxes, device, temperature=1., reduction='mean'):
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
                #print("The shape of cls toekn :", crop_feature.shape)
                #print("The shape of box_dense_features before concat cls toekn :", box_dense_features.shape)

                # concat CLS token of the box
                # shape of sim_matrix: [num of patches per box, num of patches per box + 1]
                #expanded_cls = crop_feature.expand(box_dense_features.shape[0], -1)

                student_dense_features = torch.cat([box_dense_features, crop_feature], dim=0)
                student_dense_feature = self.local_dim(student_dense_features, device)
                label = F.softmax(crop_feature, dim=-1)
                total_loss = torch.sum(-label * F.log_softmax(student_dense_features, dim=-1), dim=-1).mean()
                loss.append(total_loss.mean()*0.05)

        return sum(loss) / len(loss)

    def infonce(self, dense_features, crop_features, denormed_boxes, temperature=1., reduction='mean'):
        crop_idx = 0
        loss = []

        # if 'intra_batch_patch' in args.neg_type:
        #     intra_batch_patch_dense_features = []
        # if 'intra_batch_cls' in args.neg_type:
        #     intra_batch_grid_cls_feature = []
        if self.intra_batch:
            intra_batch_patch_dense_features = []
            pos_sims_batch = []
            

        # for all images in a batch
        for i in range(len(denormed_boxes)):
            boxes_single_image = denormed_boxes[i].round().int()

            if self.inter_grid or self.intra_batch:
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
                print("The shape of cls toekn :", crop_feature.shape)
                print("The shape of box_dense_features before concat cls toekn :", box_dense_features.shape)

                # concat CLS token of the box
                box_dense_features = torch.cat([crop_feature, box_dense_features], dim=0)
                print("The shape of box_dense_features after concat cls toekn :", box_dense_features.shape)
                # shape of sim_matrix: [num of patches per box, num of patches per box + 1]
                sim_matrix = torch.matmul(box_dense_features, box_dense_features.T).fill_diagonal_(0)[1:,:]
                print("The shape of sim_matrix :", sim_matrix)

            if self.inter_grid:
                if self.intra_batch is False:
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
                
        if self.intra_batch:
            assert len(intra_batch_patch_dense_features) == 2, f'only batch size 2 available when using intra_batch option'
            dense_features_1, dense_features_2 = intra_batch_patch_dense_features
            pos_sim_1, pos_sim_2 = pos_sims_batch
            neg_sim = torch.matmul(dense_features_1, dense_features_2.T)
            sim_matrix_1, sim_matrix_2 = torch.cat([pos_sim_1, neg_sim], axis=1), torch.cat([pos_sim_2, neg_sim.T], axis=1)
            label_1, label_2 = torch.zeros(sim_matrix_1.shape[0], dtype=torch.long, device=sim_matrix.device), torch.zeros(sim_matrix_2.shape[0], dtype=torch.long, device=sim_matrix.device)
            
            loss += [F.cross_entropy(sim_matrix_1*temperature, label_1, reduction=reduction), F.cross_entropy(sim_matrix_2*temperature, label_2, reduction=reduction)]

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
