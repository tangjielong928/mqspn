import pdb
import random

import torch
import os
from PIL import Image
from src import util

def create_vision_train_sample(img_path, region_path, mapping_region, transform= None, clip_processor=None, aux_processor=None, rcnn_processor=None,
                                                 aux_size=None, rcnn_size=None, path_num = None):
    if region_path is not None:
        # image process
        try:
            image = Image.open(region_path).convert('RGB')
            image = clip_processor(images=image, return_tensors='pt')['pixel_values'].squeeze()
        except:
            # 不存在对应图片
            img_path = os.path.join(img_path, '0.jpg')
            image = Image.open(img_path).convert('RGB')
            image = clip_processor(images=image, return_tensors='pt')['pixel_values'].squeeze()

        bboxs = mapping_region['bbox']
        # first region is for ungroundable
        aux_imgs = []
        # aux_imgs.append(torch.zeros((3, aux_size, aux_size)))
        unground_region = Image.open(os.path.join(img_path, '0.jpg')).convert('RGB')
        aux_imgs.append(aux_processor(images=unground_region, return_tensors='pt')['pixel_values'].squeeze())
        for i in range(min(path_num-1, len(bboxs))):
            aux_img = Image.open(region_path).convert('RGB')
            # 裁剪图片
            left, top, right, bottom = bboxs[i]
            aux_img = aux_img.crop((left, top, right, bottom))
            aux_img = aux_processor(images=aux_img, return_tensors='pt')['pixel_values'].squeeze()
            aux_imgs.append(aux_img)
        for i in range(path_num - len(bboxs) -1):
            aux_imgs.append(torch.zeros((3, aux_size, aux_size)))

        if len(bboxs) + 1 < path_num:
            region_mask = [1]*(len(bboxs)+1) + [0]*(path_num - len(bboxs) -1)
        else:
            region_mask = [1] * path_num
        region_mask = torch.tensor(region_mask, dtype=torch.bool)
        aux_imgs = torch.stack(aux_imgs, dim=0)
        assert len(aux_imgs) == path_num == len(region_mask)

        return image, aux_imgs, region_mask

def create_vision_train_sample_boxfea(img_path, region_path, mapping_region, transform= None, clip_processor=None, aux_processor=None, rcnn_processor=None,
                                                 aux_size=None, rcnn_size=None, path_num = None, embedding_size = 2048):
    if region_path is not None:

        bboxs = mapping_region['bbox']
        box_features = mapping_region['box_features']
        assert len(bboxs) == len(box_features)
        # first region is for ungroundable
        aux_imgs = []
        # aux_imgs.append(torch.zeros((3, aux_size, aux_size)))
        aux_imgs.append(torch.zeros(embedding_size))
        for i in range(min(path_num-1, len(bboxs))):
            aux_imgs.append(torch.tensor(box_features[i], dtype=torch.float))
        for i in range(path_num - len(bboxs) -1):
            aux_imgs.append(torch.zeros(embedding_size))

        if len(bboxs) + 1 < path_num:
            region_mask = [1]*(len(bboxs)+1) + [0]*(path_num - len(bboxs) -1)
        else:
            region_mask = [1] * path_num
        region_mask = torch.tensor(region_mask, dtype=torch.bool)
        aux_imgs = torch.stack(aux_imgs, dim=0)
        assert len(aux_imgs) == path_num

        return  None, aux_imgs, region_mask

def create_train_sample(doc, random_mask = False, tokenizer = None, repeat_gt_entities = -1,
                        transform= None, clip_processor=None, aux_processor=None, rcnn_processor=None, image_path = None, aux_size=None, rcnn_size=None, path_num = None):
    pos_encoding = [t.pos_id for t in doc.tokens]
    encodings = doc.encoding
    seg_encoding = doc.seg_encoding
    # if len(doc.encoding) > 512:
    #     return None
    token_count = len(doc.tokens)
    context_size = len(encodings)

    gt_seq_labels = [0] * len(encodings)
    special_tokens_map = tokenizer.special_tokens_map
    if random_mask:
        if random.random() < 0.5:
            for i in range(len(gt_seq_labels) -1):
                replace_rnd = random.random()
                if replace_rnd < 0.15 and i != 0:
                    gt_seq_labels[i] = encodings[i]
                    strategy_rnd = random.random()
                    if strategy_rnd < 0.8:
                        encodings[i] = tokenizer.convert_tokens_to_ids(special_tokens_map['mask_token'])
                    elif strategy_rnd < 0.9:
                        encodings[i] = random.randint(0, tokenizer.vocab_size - 1)
        # else:
        #     gt_seq_labels[0] = encodings[0]

    char_encodings = doc.char_encoding
    char_encoding = []
    char_count = []
    for char_encoding_token in char_encodings:
        char_count.append(len(char_encoding_token))
        char_encoding.append(torch.tensor(char_encoding_token,dtype=torch.long))
    char_encoding = util.padded_stack(char_encoding)
    token_masks_char = (char_encoding!=0).long()
    char_count = torch.tensor(char_count, dtype = torch.long)

    wordvec_encoding = [t.wordinx for t in doc.tokens]
    
    # import pdb; pdb.set_trace()
    # all tokens
    context2token_masks = []
    for t in doc.tokens:
        context2token_masks.append(create_entity_mask(*t.span, context_size))

    gt_entities_spans_token = []
    gt_entity_types = []
    gt_entity_masks = []
    gt_entity_regions = []

    for e in doc.entities:
        gt_entities_spans_token.append(e.span_token)
        gt_entity_types.append(e.entity_type.index)
        gt_entity_masks.append(1)
        gt_entity_regions.append(e.match_region)

    if repeat_gt_entities != -1:
        if len(doc.entities)!=0: #有实体, region index 0 is for ungroundable
            new_list = []
            for entity, type, region, mask in zip(gt_entities_spans_token, gt_entity_types, gt_entity_regions, gt_entity_masks):
                if len(region)==0:
                    new_list.append([entity, type, 0, mask])
                else:
                    for i in region:
                        if i['region_index']<path_num:
                            new_list.append([entity, type, i['region_index'], mask])
                    if len(new_list)==0:
                        new_list.append([entity, type, 0, mask])
            k = repeat_gt_entities//len(new_list)
            m = repeat_gt_entities % len(new_list)
            new_list = new_list*k + new_list[:m]
            gt_entities_spans_token = [x[0] for x in new_list]
            gt_entity_types = [x[1] for x in new_list]
            gt_entity_regions = [x[2] for x in new_list]
            gt_entity_masks = [x[3] for x in new_list]
            assert len(gt_entities_spans_token) == len(gt_entity_types) == len(gt_entity_masks) == len(gt_entity_regions) == repeat_gt_entities

    # create tensors
    # token indices
    encodings = torch.tensor(encodings, dtype=torch.long)
    seg_encoding = torch.tensor(seg_encoding, dtype=torch.long)
    gt_seq_labels = torch.tensor(gt_seq_labels, dtype=torch.long)
    pos_encoding = torch.tensor(pos_encoding, dtype=torch.long)
    wordvec_encoding = torch.tensor(wordvec_encoding, dtype=torch.long)

    # masking of tokens
    context_masks = torch.ones(context_size, dtype=torch.bool)
    # context_masks = torch.tensor(seg_encoding, dtype=torch.bool)
    token_masks = torch.ones(token_count, dtype=torch.bool)

    context2token_masks = torch.stack(context2token_masks)

    if len(gt_entity_types) > 0:
        gt_entity_types = torch.tensor(gt_entity_types, dtype=torch.long)
        # gt_entity_spans_token = torch.tensor(gt_entities_spans_token, dtype=torch.float) / len(doc.tokens)
        gt_entity_spans_token = torch.tensor(gt_entities_spans_token, dtype=torch.long)
        # gt_entity_spans_token[:, 1] = gt_entity_spans_token[:, 1] - 1
        gt_entity_masks = torch.tensor(gt_entity_masks, dtype=torch.bool)
        gt_entity_regions = torch.tensor(gt_entity_regions, dtype=torch.long)
    else:
        gt_entity_types = torch.zeros([1], dtype=torch.long)
        gt_entity_spans_token = torch.zeros([1, 2], dtype=torch.long)
        gt_entity_masks = torch.zeros([1], dtype=torch.bool)
        gt_entity_regions = torch.zeros([1], dtype=torch.long)

    img_path = doc.regions.image_path
    region_path = doc.regions.region_path
    mapping_region = doc.regions.mapping_region
    image, aux_imgs, region_mask = create_vision_train_sample(img_path, region_path, mapping_region,transform= transform, clip_processor=clip_processor, aux_processor=aux_processor, rcnn_processor=rcnn_processor,
                                                 aux_size=aux_size, rcnn_size=rcnn_size, path_num = path_num)
    # image, aux_imgs, region_mask = create_vision_train_sample_boxfea(img_path, region_path, mapping_region, path_num = path_num)
    return dict(encodings=encodings, context_masks=context_masks, seg_encoding = seg_encoding, context2token_masks=context2token_masks, token_masks=token_masks, 
                pos_encoding = pos_encoding, wordvec_encoding = wordvec_encoding, char_encoding = char_encoding, token_masks_char = token_masks_char, char_count = char_count,
                gt_types=gt_entity_types, gt_spans=gt_entity_spans_token, entity_masks=gt_entity_masks, gt_seq_labels = gt_seq_labels, meta_doc = doc,
                gt_entity_regions = gt_entity_regions, image = image, aux_imgs = aux_imgs, region_mask = region_mask)


def create_eval_sample(doc,transform= None, clip_processor=None, aux_processor=None, rcnn_processor=None, image_path = None, aux_size=None, rcnn_size=None, path_num = None):
    # if len(doc.encoding) > 512:
    #     return None
    pos_encoding = [t.pos_id for t in doc.tokens]
    encodings = doc.encoding
    seg_encoding = doc.seg_encoding
    token_count = len(doc.tokens)
    context_size = len(encodings)

    char_encodings = doc.char_encoding
    char_encoding = []
    char_count = []
    for char_encoding_token in char_encodings:
        char_count.append(len(char_encoding_token))
        char_encoding.append(torch.tensor(char_encoding_token,dtype=torch.long))
    char_encoding = util.padded_stack(char_encoding)
    token_masks_char = (char_encoding!=0).long()
    char_count = torch.tensor(char_count, dtype = torch.long)

    wordvec_encoding = [t.wordinx for t in doc.tokens]
    
    # import pdb; pdb.set_trace()
    # all tokens
    context2token_masks = []
    for t in doc.tokens:
        context2token_masks.append(create_entity_mask(*t.span, context_size))

    # create tensors
    # token indices
    encodings = torch.tensor(encodings, dtype=torch.long)
    seg_encoding = torch.tensor(seg_encoding, dtype=torch.long)
    
    pos_encoding = torch.tensor(pos_encoding, dtype=torch.long)
    wordvec_encoding = torch.tensor(wordvec_encoding, dtype=torch.long)

    # masking of tokens
    context_masks = torch.ones(context_size, dtype=torch.bool)
    token_masks = torch.ones(token_count, dtype=torch.bool)

    # also create samples_masks:
    # tensors to mask entity/relation samples of batch
    # since samples are stacked into batches, "padding" entities/relations possibly must be created
    # these are later masked during loss computation
    context2token_masks = torch.stack(context2token_masks)

    # vision part
    img_path = doc.regions.image_path
    region_path = doc.regions.region_path
    mapping_region = doc.regions.mapping_region
    image, aux_imgs, region_mask = create_vision_train_sample(img_path, region_path, mapping_region, transform=transform,
                                                 clip_processor=clip_processor, aux_processor=aux_processor,
                                                 rcnn_processor=rcnn_processor,
                                                 aux_size=aux_size, rcnn_size=rcnn_size, path_num=path_num)
    # image, aux_imgs, region_mask = create_vision_train_sample_boxfea(img_path, region_path, mapping_region, path_num=path_num)
    return dict(encodings=encodings, context_masks=context_masks, seg_encoding = seg_encoding, context2token_masks=context2token_masks, token_masks=token_masks, 
                pos_encoding = pos_encoding, wordvec_encoding = wordvec_encoding, char_encoding = char_encoding, token_masks_char = token_masks_char,
                char_count = char_count, meta_doc = doc,image = image, aux_imgs = aux_imgs, region_mask = region_mask)

def create_entity_mask(start, end, context_size):
    mask = torch.zeros(context_size, dtype=torch.bool)
    mask[start:end] = 1
    return mask

def collate_fn_padding(batch):
    batch = list(filter(lambda x: x is not None, batch))
    padded_batch = dict()
    keys = batch[0].keys()

    for key in keys:
        samples = [s[key] for s in batch]
        if key.startswith("meta"):
            padded_batch[key] = samples
            continue

        if not batch[0][key].shape:
            padded_batch[key] = torch.stack(samples)
        else:
            padded_batch[key] = util.padded_stack([s[key] for s in batch])

    return padded_batch
