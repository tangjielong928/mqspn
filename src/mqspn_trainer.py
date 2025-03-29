from builtins import sum
import argparse
from collections import defaultdict
import json
import math
import os
import gc

import torch
from torch.nn import DataParallel
from torch.optim import Optimizer
from torchvision import transforms
from torch.utils.data import DataLoader
from src.vision_models.modeling_clip import CLIPModel
from transformers import AdamW, BertConfig, CLIPConfig, BertModel, CLIPProcessor
from transformers import BertTokenizer, RobertaTokenizer, AutoTokenizer

from src import models
from src import sampling
from src import util
from src.entities import Dataset, DistributedIterableDataset
from src.evaluator import Evaluator
from src.input_reader import JsonInputReader, BaseInputReader
from src.loss import MQSPNLoss, Loss
from tqdm import tqdm
from src.trainer import BaseTrainer
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))


def get_linear_schedule_with_warmup_two_stage(optimizer, num_warmup_steps_stage_one, num_training_steps_stage_one,
                                              num_warmup_steps_stage_two, num_training_steps_stage_two,  last_epoch=-1):
    def lr_lambda(current_step: int):
        if current_step < num_training_steps_stage_one:
            if current_step < num_warmup_steps_stage_one:
                return float(current_step) / float(max(1, num_warmup_steps_stage_one))
            return max(
                0.0, float(num_training_steps_stage_one - current_step) / float(max(1, num_training_steps_stage_one - num_warmup_steps_stage_one))
            )
        else:
            current_step = current_step - num_training_steps_stage_one
            if current_step < num_warmup_steps_stage_two:
                return float(current_step) / float(max(1, num_warmup_steps_stage_two))
            return max(
                0.0, float(num_training_steps_stage_two - current_step) / float(max(1, num_training_steps_stage_two - num_warmup_steps_stage_two))
            )
    return LambdaLR(optimizer, lr_lambda, last_epoch)

class MQSPNTrainer(BaseTrainer):
    """ GMNER training and evaluation """

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        # text tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path,
                                                    local_files_only = True,
                                                    use_fast = False,
                                                    do_lower_case=args.lowercase,
                                                    cache_dir=args.cache_path)
        # vision
        # 组合图像预处理
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
        self.clip_processor = CLIPProcessor.from_pretrained(args.vit_name)
        self.aux_processor = CLIPProcessor.from_pretrained(args.vit_name)
        self.aux_processor.feature_extractor.size, self.aux_processor.feature_extractor.crop_size = args.aux_size, args.aux_size
        self.rcnn_processor = CLIPProcessor.from_pretrained(args.vit_name)
        self.rcnn_processor.feature_extractor.size, self.rcnn_processor.feature_extractor.crop_size = args.rcnn_size, args.rcnn_size
        self.clip_model = CLIPModel.from_pretrained(args.vit_name)
        self.clip_vit = self.clip_model.vision_model
        self.clip_bert = self.clip_model.text_model
        self.aux_size = args.aux_size
        self.rcnn_size = args.rcnn_size
        self.image_path = args.image_path
        self.path_num = args.path_num
        self._logger.info("clip-vit loading success!!")

        # path to export predictions to
        self._predictions_path = os.path.join(self._log_path, 'predictions_%s_epoch_%s.json')

        # path to export relation extraction examples to
        self._examples_path = os.path.join(self._log_path, 'examples_%s_%s_epoch_%s.html')

        self._logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

    def load_model(self, input_reader, is_eval = False):
        args = self.args


        # create model
        model_class = models.get_model(args.model_type)
        vision_config = CLIPConfig.from_pretrained(args.vit_name).vision_config
        bert = BertModel.from_pretrained(args.model_path)
        # load model
        # config = BertConfig.from_pretrained(args.model_path, cache_dir=args.cache_path)
        text_config = models.EntityAwareBertConfig.from_pretrained(args.model_path, cache_dir=args.cache_path, entity_queries_num = args.entity_queries_num, entity_emb_size = args.entity_emb_size,
                                              mask_ent2tok = args.mask_ent2tok,  mask_tok2ent = args.mask_tok2ent, mask_ent2ent = args.mask_ent2ent,
                                              mask_entself = args.mask_entself, entity_aware_attention = args.entity_aware_attention,
                                              entity_aware_selfout = args.entity_aware_selfout, entity_aware_intermediate = args.entity_aware_intermediate,
                                              entity_aware_output = args.entity_aware_output, use_entity_pos = args.use_entity_pos,
                                              use_entity_common_embedding = args.use_entity_common_embedding)

        embed = None
        if args.use_glove:
            embed = torch.from_numpy(input_reader.embedding_weight).float()
        model = model_class(# proxies = {'http': '10.15.82.42:7890'},
                            # local_files_only = True,
                            text_config = text_config,
                            vision_config = vision_config,
                            embed = embed,
                            entity_type_count=input_reader.entity_type_count,
                            prop_drop=args.prop_drop,
                            freeze_transformer=args.freeze_transformer,
                            pos_size = args.pos_size,
                            char_lstm_layers = args.char_lstm_layers,
                            char_lstm_drop = args.char_lstm_drop,
                            char_size = args.char_size,
                            use_glove = args.use_glove,
                            use_pos = args.use_pos,
                            use_char_lstm = args.use_char_lstm,
                            lstm_layers = args.lstm_layers,
                            pool_type = args.pool_type,
                            word_mask_tok2ent = args.word_mask_tok2ent,
                            word_mask_ent2tok = args.word_mask_ent2tok,
                            word_mask_ent2ent = args.word_mask_ent2ent,
                            word_mask_entself = args.word_mask_entself,
                            share_query_pos = args.share_query_pos,
                            use_token_level_encoder = args.use_token_level_encoder,
                            num_token_entity_encoderlayer = args.num_token_entity_encoderlayer,
                            use_entity_attention = args.use_entity_attention,
                            use_masked_lm = args.use_masked_lm,
                            use_aux_loss =  args.use_aux_loss,
                            use_lstm = args.use_lstm,
                            inlcude_subword_aux_loss = args.inlcude_subword_aux_loss,
                            last_layer_for_loss = args.last_layer_for_loss,
                            split_epoch = args.split_epoch)

        model_dict = model.state_dict()
        bert_model_dict = bert.state_dict()
        clip_model_dict = self.clip_vit.state_dict()
        vision_names, text_names = [], []
        for name in model_dict:
            if 'vision' in name:
                clip_name = name.replace('vision_', '').replace('model.', '').replace('bert.', '')
                if clip_name in clip_model_dict:
                    vision_names.append(clip_name)
                    model_dict[name] = clip_model_dict[clip_name]
            else:
                text_name = name.replace('bert.', '').replace('model.', '')
                if text_name in bert_model_dict:
                    text_names.append(text_name)
                    model_dict[name] = bert_model_dict[text_name]
        model.load_state_dict(model_dict)
        num_params = sum(param.numel() for param in model.parameters())
        self._logger.info(f"Model Parameters Number: {num_params}")

        # eval mode
        if is_eval:
            # 拿到保存的参数
            model_static_dict = torch.load(args.checkpoint_path)
            # 把参数加载到模型中
            model.load_state_dict(model_static_dict)

        # if not is_eval and args.copy_weight:
        #     with torch.no_grad():
        #         state_dict = None
        #         state_dict = model.model.state_dict().copy()
        #         for name, param in state_dict.items():
        #             countpart = name.replace("entity_", "").replace("e2w_","")
        #             if countpart not in state_dict or countpart == name or "embedding" in name or "LayerNorm" in name:
        #                 # print(countpart)
        #                 continue
        #             print(f"copy {countpart} to {name}")
        #             param.copy_(state_dict[countpart])

        return model

    def train(self, train_path: str, valid_path: str, types_path: str, input_reader_cls: BaseInputReader):
        args = self.args
        train_label, valid_label = 'train', 'valid'

        if self.record:
            self._logger.info("Datasets: %s, %s" % (train_path, valid_path))
            self._logger.info("Model type: %s" % args.model_type)

            # create log csv files
            self._init_train_logging(train_label)

        # read datasets
        input_reader = input_reader_cls(types_path,args.xml_path, args.detection_path, self._tokenizer, self._logger, wordvec_filename = args.wordvec_path, random_mask_word = args.use_masked_lm,
                                        use_glove = args.use_glove, use_pos = args.use_pos, repeat_gt_entities = args.repeat_gt_entities, candidate_num = args.candidate_regions_num,
                                        transform= self.transform, clip_processor=self.clip_processor, aux_processor=self.aux_processor,
                                        rcnn_processor=self.rcnn_processor, image_path = self.image_path, aux_size=self.aux_size, rcnn_size=self.rcnn_size, path_num = self.path_num)
        input_reader.read({train_label: train_path, valid_label: valid_path})

        if self.local_rank < 1:
            self._log_datasets(input_reader)

        world_size = 1
        if args.local_rank != -1:
            world_size = dist.get_world_size()

        train_dataset = input_reader.get_dataset(train_label)
        train_sample_count = train_dataset.document_count
        updates_epoch = train_sample_count // (args.train_batch_size * world_size)
        updates_total_stage_one = updates_epoch * args.split_epoch
        updates_total_stage_two = updates_epoch * (args.epochs - args.split_epoch)

        validation_dataset = input_reader.get_dataset(valid_label)

        if self.record:
            self._logger.info("Updates per epoch: %s" % updates_epoch)
            self._logger.info("Updates total: %s" % (updates_total_stage_one + updates_total_stage_two))

        model = self.load_model(input_reader, is_eval = False)

        model.to(self._device)
        if args.local_rank != -1:
            model = DDP(model, device_ids=[args.local_rank], find_unused_parameters=False)

        # create optimizer
        optimizer_params = self._get_optimizer_params(model)
        optimizer = AdamW(optimizer_params, lr=args.lr, weight_decay=args.weight_decay, correct_bias=False)
        scheduler = get_linear_schedule_with_warmup_two_stage(optimizer,
                                                            num_warmup_steps_stage_one = args.lr_warmup * updates_total_stage_one,
                                                            num_training_steps_stage_one = updates_total_stage_one,
                                                            num_warmup_steps_stage_two = args.lr_warmup * updates_total_stage_two,
                                                            num_training_steps_stage_two = updates_total_stage_two)


        compute_loss = MQSPNLoss(input_reader.entity_type_count, self._device, model, optimizer, scheduler, args.max_grad_norm, args.nil_weight, args.match_class_weight, args.match_boundary_weight, args.match_region_weight, args.loss_class_weight, args.loss_boundary_weight, args.loss_region_weight, args.type_loss, solver = args.match_solver, match_warmup_epoch = args.match_warmup_epoch)

        # eval validation set
        if args.init_eval and self.record:
            self._eval(model, validation_dataset, input_reader, 0, updates_epoch)

        # train
        best_f1 = 0
        best_epoch = 0
        for epoch in range(args.epochs):
            if epoch == args.split_epoch:
                optimizer.__setstate__({'state': defaultdict(dict)})
            # train epoch
            self._train_epoch(model, compute_loss, optimizer, train_dataset, updates_epoch, epoch)

            # eval validation sets
            if (not args.final_eval or (epoch == args.epochs - 1)) and self.record:
                f1 = self._eval(model, validation_dataset, input_reader, epoch + 1, updates_epoch)
                # self._save_best(model, self._tokenizer,optimizer if args.save_optimizer else None,f1[2],epoch * updates_epoch, "best")
                if best_f1 < f1[5]:
                    self._logger.info(f"Best GMNER F1 score update, from {best_f1} to {f1[5]}")
                    best_f1 = f1[5]
                    best_epoch = epoch + 1
                    extra = dict(epoch=epoch, updates_epoch=updates_epoch, epoch_iteration=0)
                    # if "pretrain" in args.label:
                    self._save_model(self._save_path, model, self._tokenizer, epoch * updates_epoch,
                        optimizer=optimizer if args.save_optimizer else None, extra=extra,
                        include_iteration=False, name='best_model')
            if self.record:
                if args.save_path_include_iteration:
                    self._save_model(self._save_path, model, self._tokenizer, epoch,
                            optimizer=optimizer if args.save_optimizer else None, extra=extra,
                            include_iteration=args.save_path_include_iteration, name='model')
                self._logger.info(f"Best gmner F1 score: {best_f1}, achieved at Epoch: {best_epoch}")

        # save final model
        extra = dict(epoch=args.epochs, updates_epoch=updates_epoch, epoch_iteration=0)
        global_iteration = args.epochs * updates_epoch
        if self.record:
            self._save_model(self._save_path, model, self._tokenizer, global_iteration,
                            optimizer=optimizer if args.save_optimizer else None, extra=extra,
                            include_iteration=False, name='final_model')
            self._logger.info("Logged in: %s" % self._log_path)
            self._logger.info("Saved in: %s" % self._save_path)
            self._close_summary_writer()

    def eval(self, dataset_path: str, types_path: str, input_reader_cls: BaseInputReader):
        args = self.args
        dataset_label = 'test'

        self._logger.info("Dataset: %s" % dataset_path)
        self._logger.info("Model: %s" % args.model_type)


        # read datasets
        input_reader = input_reader_cls(types_path, args.xml_path, args.detection_path, self._tokenizer, self._logger, wordvec_filename=args.wordvec_path,
                                        random_mask_word=args.use_masked_lm,
                                        use_glove=args.use_glove, use_pos=args.use_pos,
                                        repeat_gt_entities=args.repeat_gt_entities,
                                        transform=self.transform, candidate_num = args.candidate_regions_num, clip_processor=self.clip_processor,
                                        aux_processor=self.aux_processor,
                                        rcnn_processor=self.rcnn_processor, image_path=self.image_path,
                                        aux_size=self.aux_size, rcnn_size=self.rcnn_size, path_num=self.path_num)
        input_reader.read({dataset_label: dataset_path})
        # self._log_datasets(input_reader)

        model = self.load_model(input_reader, is_eval = True)

        model.to(self._device)
        # if args.local_rank != -1:
        #     model = DDP(model, device_ids=[args.local_rank])


        # evaluate
        self._eval(model, input_reader.get_dataset(dataset_label), input_reader)

        self._logger.info("Logged in: %s" % self._log_path)
        self._close_summary_writer()

    def _train_epoch(self, model: torch.nn.Module, compute_loss: Loss, optimizer: Optimizer, dataset,
                     updates_epoch: int, epoch: int):
        args = self.args
        self._logger.info("Train epoch: %s" % epoch)

        # create data loader
        dataset.switch_mode(Dataset.TRAIN_MODE)

        word_size = 1
        if args.local_rank != -1:
            word_size = dist.get_world_size()

        train_sampler = None
        shuffle = False
        if isinstance(dataset, Dataset):
            if len(dataset) < 100000:
                shuffle = True
            if args.local_rank != -1:
                train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas = word_size,rank = args.local_rank, shuffle = shuffle)
                shuffle = False

        data_loader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=shuffle, drop_last=True,
                                    num_workers=args.sampling_processes, collate_fn=sampling.collate_fn_padding,  sampler=train_sampler)


        model.zero_grad()

        iteration = 0
        total = math.ceil((dataset.document_count // args.train_batch_size) / word_size)
        for batch in tqdm(data_loader, total=total, desc='Train epoch %s' % epoch):
            model.train()
            # print(batch['encodings'].size)
            batch = util.to_device(batch, self._device)


            # forward step 
            entity_logits, p_left, p_right, p_region, masked_seq_logits, output = model(encodings=batch['encodings'], context_masks=batch['context_masks'], seg_encoding = batch['seg_encoding'],
                                                                              context2token_masks=batch['context2token_masks'], token_masks=batch['token_masks'], epoch = epoch,
                                                                              pos_encoding = batch['pos_encoding'], wordvec_encoding = batch['wordvec_encoding'], char_encoding = batch['char_encoding'],
                                                                              token_masks_char = batch['token_masks_char'], char_count = batch['char_count'],
                                                                              image = batch['image'], aux_imgs = batch['aux_imgs'], region_mask = batch['region_mask'])

            # compute loss and optimize parameters
            batch_loss = compute_loss.compute(entity_logits, p_left, p_right, p_region, output, gt_region = batch['gt_entity_regions'],gt_types=batch['gt_types'], gt_spans = batch['gt_spans'], entity_masks=batch['entity_masks'],
                                              epoch = epoch,  deeply_weight = args.deeply_weight, seq_logits = masked_seq_logits, gt_seq_labels=batch['gt_seq_labels'], batch = batch)

            # logging
            iteration += 1
            global_iteration = epoch * updates_epoch + iteration

            if global_iteration % args.train_log_iter == 0 and self.local_rank < 1:
                self._log_train(optimizer, batch_loss, epoch, iteration, global_iteration, dataset.label)

        return iteration

    def _eval(self, model: torch.nn.Module, dataset, input_reader: JsonInputReader,
              epoch: int = 0, updates_epoch: int = 0, iteration: int = 0):
        args = self.args
        self._logger.info("Evaluate: %s" % dataset.label)

        # if isinstance(model, DataParallel):
        #     # currently no multi GPU support during evaluation
        #     model = model.module

        # create evaluator
        evaluator = Evaluator(dataset, input_reader, self._tokenizer, self._logger, args.no_overlapping, args.no_partial_overlapping, args.no_duplicate, self._predictions_path,
                              self._examples_path, args.example_count, epoch, dataset.label, cls_threshold = args.cls_threshold, boundary_threshold = args.boundary_threshold,
                              region_threshold = args.region_threshold,save_prediction = args.store_predictions, path_num = args.path_num)

        # create data loader
        dataset.switch_mode(Dataset.EVAL_MODE)

        word_size = 1
        eval_sampler = None

        if isinstance(dataset, Dataset):
            data_loader = DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False, drop_last=False,
                                 num_workers=args.sampling_processes, collate_fn=sampling.collate_fn_padding, sampler=eval_sampler)
        else:
            data_loader = DataLoader(dataset, batch_size=args.eval_batch_size, drop_last=False, collate_fn=sampling.collate_fn_padding, sampler=eval_sampler)

        with torch.no_grad():
            model.eval()

            # iterate batches
            total = math.ceil(dataset.document_count / (args.eval_batch_size * word_size))
            for batch in tqdm(data_loader, total=total, desc='Evaluate epoch %s' % epoch):
                # move batch to selected device
                batch = util.to_device(batch, self._device)

                # run model (forward pass)
                entity_logits, p_left, p_right, p_region, _, outputs = model(encodings=batch['encodings'], context_masks=batch['context_masks'], seg_encoding = batch['seg_encoding'], context2token_masks=batch['context2token_masks'], token_masks=batch['token_masks'], pos_encoding = batch['pos_encoding'], wordvec_encoding = batch['wordvec_encoding'], char_encoding = batch['char_encoding'], token_masks_char = batch['token_masks_char'], char_count = batch['char_count'], evaluate = True,
                                                                   image = batch['image'], aux_imgs = batch['aux_imgs'], region_mask = batch['region_mask'])

                # evaluate batch
                evaluator.eval_batch(entity_logits, p_left, p_right, p_region, outputs, batch)
        global_iteration = epoch * updates_epoch + iteration
        gmner_eval = evaluator.compute_scores()

        # self.scheduler.step(ner_eval[2])
        if args.store_predictions:
            evaluator.store_predictions()

        if args.store_examples:
            evaluator.store_examples()

        return gmner_eval

    def _get_optimizer_params(self, model):
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        # regressier
        optimizer_params = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

        return optimizer_params

    def _log_train(self, optimizer: Optimizer, loss: float, epoch: int,
                   iteration: int, global_iteration: int, label: str):
        # average loss
        avg_loss = loss / self.args.train_batch_size
        # get current learning rate
        lr = self._get_lr(optimizer)[0]

        # log to tensorboard
        self._log_tensorboard(label, 'loss', loss, global_iteration)
        self._log_tensorboard(label, 'loss_avg', avg_loss, global_iteration)
        self._log_tensorboard(label, 'lr', lr, global_iteration)

        # log to csv
        self._log_csv(label, 'loss', loss, epoch, iteration, global_iteration)
        self._log_csv(label, 'loss_avg', avg_loss, epoch, iteration, global_iteration)
        self._log_csv(label, 'lr', lr, epoch, iteration, global_iteration)

    def _log_datasets(self, input_reader):
        # self._logger.info("Relation type count: %s" % input_reader.relation_type_count)
        self._logger.info("Entity type count: %s" % input_reader.entity_type_count)

        self._logger.info("Entities:")
        for e in input_reader.entity_types.values():
            self._logger.info(e.verbose_name + '=' + str(e.index))

        for k, d in input_reader.datasets.items():
            self._logger.info('Dataset: %s' % k)
            self._logger.info("Document count: %s" % d.document_count)
            # self._logger.info("Relation count: %s" % d.relation_count)
            self._logger.info("Entity count: %s" % d.entity_count)

        self._logger.info("Context size: %s" % input_reader.context_size)

    def _init_train_logging(self, label):
        self._add_dataset_logging(label,
                                  data={'lr': ['lr', 'epoch', 'iteration', 'global_iteration'],
                                        'loss': ['loss', 'epoch', 'iteration', 'global_iteration'],
                                        'loss_avg': ['loss_avg', 'epoch', 'iteration', 'global_iteration']})
