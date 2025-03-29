from codecs import encode
import json
from abc import abstractmethod, ABC
from collections import OrderedDict
from logging import Logger
import os
from pdb import set_trace
import tokenize
from typing import Iterable, List
import numpy as np
import string
from tqdm import tqdm
from transformers import BertTokenizer
import transformers
from tqdm import tqdm
import os
import torch
import torchvision
from PIL import Image
import xml.etree.ElementTree as ET
import numpy as np
import cv2
from src import util
from src.entities import Dataset, EntityType, RelationType, Entity, Relation, Document, DistributedIterableDataset, Region
from collections import Counter


class BaseInputReader(ABC):
    def __init__(self, types_path: str, xml_path: str, detection_path: str, tokenizer: BertTokenizer, logger: Logger = None, random_mask_word = None, repeat_gt_entities = None,
                 transform= None, candidate_num = None, clip_processor=None, aux_processor=None, rcnn_processor=None, image_path = None, aux_size=None, rcnn_size=None, path_num = None):

        types = json.load(open(types_path), object_pairs_hook=OrderedDict, )  # entity + relation types
        self._entity_types = OrderedDict()
        self._idx2entity_type = OrderedDict()
        self._relation_types = OrderedDict()
        self._idx2relation_type = OrderedDict()

        # vision
        self.clip_processor = clip_processor
        self.aux_processor = aux_processor
        self.rcnn_processor = rcnn_processor
        self.candidate_num = candidate_num
        self.aux_size = aux_size
        self.rcnn_size = rcnn_size
        self.transform = transform
        self.image_path = image_path
        self.path_num = path_num
        self.xml_path = xml_path
        self.detection_path = detection_path

        # entities
        # add 'None' entity type
        none_entity_type = EntityType('None', 0, 'None', 'No Entity')
        self._entity_types['None'] = none_entity_type
        self._idx2entity_type[0] = none_entity_type

        # specified entity types
        for i, (key, v) in enumerate(types['entities'].items()):
            entity_type = EntityType(key, i + 1, v['short'], v['verbose'])
            self._entity_types[key] = entity_type
            self._idx2entity_type[i + 1] = entity_type

        # relations
        # add 'None' relation type
        none_relation_type = RelationType('None', 0, 'None', 'No Relation')
        self._relation_types['None'] = none_relation_type
        self._idx2relation_type[0] = none_relation_type

        # specified relation types
        for i, (key, v) in enumerate(types['relations'].items()):
            relation_type = RelationType(key, i + 1, v['short'], v['verbose'], v['symmetric'])
            self._relation_types[key] = relation_type
            self._idx2relation_type[i + 1] = relation_type
            
        self._datasets = dict()

        self._tokenizer = tokenizer
        self._logger = logger
        self._random_mask_word = random_mask_word
        self._repeat_gt_entities = repeat_gt_entities

        self._vocabulary_size = tokenizer.vocab_size
        self._context_size = -1

    @abstractmethod
    def read(self, datasets):
        pass

    def get_dataset(self, label):
        return self._datasets[label]

    def get_entity_type(self, idx) -> EntityType:
        entity = self._idx2entity_type[idx]
        return entity

    def get_relation_type(self, idx) -> RelationType:
        relation = self._idx2relation_type[idx]
        return relation

    def _calc_context_size(self, datasets):
        sizes = [-1]

        for dataset in datasets:
            if isinstance(dataset, Dataset):
                for doc in dataset.documents:
                    sizes.append(len(doc.encoding))

        context_size = max(sizes)
        return context_size

    def _log(self, text):
        if self._logger is not None:
            self._logger.info(text)

    @property
    def datasets(self):
        return self._datasets

    @property
    def entity_types(self):
        return self._entity_types

    @property
    def relation_types(self):
        return self._relation_types

    @property
    def relation_type_count(self):
        return len(self._relation_types)

    @property
    def entity_type_count(self):
        return len(self._entity_types)

    @property
    def vocabulary_size(self):
        return self._vocabulary_size

    @property
    def context_size(self):
        return self._context_size

    def __str__(self):
        string = ""
        for dataset in self._datasets.values():
            string += "Dataset: %s\n" % dataset
            string += str(dataset)

        return string

    def __repr__(self):
        return self.__str__()


class JsonInputReader(BaseInputReader):
    def __init__(self, types_path: str, xml_path: str, detection_path: str, tokenizer: BertTokenizer, logger: Logger = None, wordvec_filename = None, random_mask_word = False,
                 use_glove = False, use_pos = False, repeat_gt_entities = None, candidate_num = None,
                 transform= None, clip_processor=None, aux_processor=None, rcnn_processor=None, image_path = None, aux_size=None, rcnn_size=None, path_num = None):
        super().__init__(types_path, xml_path, detection_path, tokenizer, logger, random_mask_word, repeat_gt_entities)
        if use_glove:
            if "glove" in wordvec_filename:
                vec_size = wordvec_filename.split(".")[-2] # str: 300d
            else:
                vec_size = "bio"
            if os.path.exists(os.path.dirname(types_path)+f"/vocab_{vec_size}.json") and os.path.exists(os.path.dirname(types_path)+f"/vocab_embed_{vec_size}.npy") :
                self._log(f"Reused vocab and word embedding from {os.path.dirname(types_path)}")
                self.build_vocab = False
                self.word2inx = json.load(open(os.path.dirname(types_path)+f"/vocab_{vec_size}.json","r"))
                self.embedding_weight = np.load(os.path.dirname(types_path)+f"/vocab_embed_{vec_size}.npy")
            else:
                self._log("Need some time to construct vocab...")
                self.word2inx = {"<unk>": 0}
                self.embedding_weight = None
                self.build_vocab = True
            self.vec_size = vec_size
        else:
            self.word2inx = {"<unk>": 0}
            self.embedding_weight = None
            self.build_vocab = False
        
        self.wordvec_filename = wordvec_filename
        self.POS_MAP = ["<UNK>"]
        if use_pos:
            for k, v in json.load(open(types_path.replace("types", "pos"))).items():
                if v > 15:
                    self.POS_MAP.append(k)

        # vision
        self.clip_processor = clip_processor
        self.aux_processor = aux_processor
        self.rcnn_processor = rcnn_processor
        self.candidate_num = candidate_num
        self.aux_size = aux_size
        self.rcnn_size = rcnn_size
        self.transform = transform
        self.image_path = image_path
        self.path_num = path_num
        self.xml_path = xml_path
        self.detection_path = detection_path
        self.mapping_regions = self._parse_regions(candidate_num)

    def load_wordvec(self, filename):
        self.embedding_weight = np.random.rand(len(self.word2inx),len(next(iter(self.word2vec.values()))))
        for word, inx in self.word2inx.items():
            if word in self.word2vec:
                self.embedding_weight[inx,:] = self.word2vec[word]

    def read(self, dataset_paths):
        for dataset_label, dataset_path in dataset_paths.items():
            if dataset_path.endswith(".jsonl"):
                assert not self.build_vocab, "forbidden build vocab for large dataset!"
                dataset = DistributedIterableDataset(dataset_label, dataset_path, self._relation_types, self._entity_types, self, random_mask_word = self._random_mask_word, tokenizer = self._tokenizer, repeat_gt_entities = self._repeat_gt_entities)
                self._datasets[dataset_label] = dataset
            else:
                dataset = Dataset(dataset_label, self._relation_types, self._entity_types, random_mask_word = self._random_mask_word,
                                  tokenizer = self._tokenizer, repeat_gt_entities = self._repeat_gt_entities,
                                  transform= self.transform, clip_processor=self.clip_processor, aux_processor=self.aux_processor, rcnn_processor=self.rcnn_processor,
                                  image_path = self.image_path, aux_size=self.aux_size, rcnn_size=self.rcnn_size, path_num = self.path_num)
                self._parse_dataset(dataset_path, dataset, dataset_label)
                self._datasets[dataset_label] = dataset

        if self.build_vocab:
            dataset_dir = os.path.dirname(next(iter(dataset_paths.values()))) 
            json.dump(self.word2inx,open(dataset_dir+f"/vocab_{self.vec_size}.json","w"))
            self.load_wordvec(self.wordvec_filename)
            np.save(dataset_dir+f"/vocab_embed_{self.vec_size}.npy",self.embedding_weight)
            self._log(f"Vocab and word embeddings cached in {dataset_dir}")
        # 计算最大sequence长度
        self._context_size = self._calc_context_size(self._datasets.values())

    def _parse_dataset(self, dataset_path, dataset, dataset_label):
        if dataset_path.endswith(".json"):
            documents = json.load(open(dataset_path))
        else:
            documents = self._parse_rawtxt(dataset_path)
        if dataset_label == "train" and self.build_vocab:
            self._build_vocab(documents)
        for document in tqdm(documents, desc="Parse dataset '%s'" % dataset.label):
            self._parse_document(document, dataset)
    
    def _build_vocab(self, documents, min_freq = 1):
        self.word2vec = {}
        with open(self.wordvec_filename, "r") as f:
            if "glove" not in self.wordvec_filename:
                f.readline()
            for line in f:
                fields = line.strip().split(' ')
                self.word2vec[fields[0]] = list(float(x) for x in fields[1:])
        counter = Counter()
        for doc in documents:
            counter.update(list(map(lambda x: x.lower(), doc['tokens'])))
        for k, v in counter.items():
            if v >= min_freq and k in self.word2vec:
                self.word2inx[k] = len(self.word2inx)

    def _parse_rawtxt(self, data_path):
        # mapping_regions = mapping_region()
        with open(data_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            raw_words, raw_targets = [], []
            raw_word, raw_target = [], []
            imgs = []
            for line in lines:
                if line.startswith("IMGID:"):
                    img_id = line.strip().split('IMGID:')[1] + '.jpg'
                    imgs.append(img_id)
                    continue
                if line != "\n":
                    raw_word.append(line.split('\t')[0])
                    label = line.split('\t')[1][:-1]
                    if 'OTHER' in label:
                        label = label[:2] + 'MISC'
                    raw_target.append(label)
                else:
                    raw_words.append(raw_word)
                    raw_targets.append(raw_target)
                    raw_word, raw_target = [], []

        assert len(raw_words) == len(raw_targets) == len(imgs), "{}, {}, {}".format(len(raw_words), len(raw_targets),
                                                                                    len(imgs))
        res = []
        label_set = set()
        for sentence, target, img_id in zip(raw_words, raw_targets, imgs):
            entities = []
            start, end = 0, 0
            while start < len(target) and end < len(target):
                if 'B-' in target[start]:
                    end = start
                    end += 1
                    while end < len(target) and 'I-' in target[end]:
                        end += 1
                    entities.append({
                        "start": start,
                        "end": end,
                        "type": target[start][2:],
                        "entity_span": sentence[start:end]
                    })
                    label_set.add(target[start][2:])
                    start = end
                start += 1
            res.append({
                "tokens": sentence,
                "relations": [],
                "pos": [],
                "ltokens": [],
                "rtokens": [],
                "img_id": img_id,
                "entities": entities,
                "mapping_region": self.mapping_regions[img_id.split('.')[0]]
            })
        return res

    def _parse_document(self, doc, dataset) -> Document:
        jtokens = doc['tokens']
        jrelations = doc['relations']
        jentities = doc['entities']
        jpos = None
        if 'pos' in doc:
            jpos = doc['pos']
        ltokens = doc["ltokens"]
        rtokens = doc["rtokens"]

        if not jpos:
            jpos = ["<UNK>"] * len(doc['tokens'])

        # vision, parse regions
        mapping_region = doc['mapping_region']
        image_id = doc['img_id']
        image_path = self.image_path
        regions = dataset.create_region(image_path, image_id, mapping_region)


        # parse tokens
        doc_tokens, doc_encoding, char_encoding, seg_encoding = self._parse_tokens(jtokens, ltokens, rtokens, jpos, dataset)

        if len(doc_encoding) > 512:
            self._log(f"Document {doc['org_id']} len(doc_encoding) = {len(doc_encoding) } > 512, Ignored!")
            return None

        # parse entity mentions
        entities = self._parse_entities(jentities, doc_tokens, dataset, mapping_region)

        # parse relations
        relations = self._parse_relations(jrelations, entities, dataset)

        # create document
        document = dataset.create_document(doc_tokens, entities, relations, doc_encoding, char_encoding, seg_encoding, regions)

        return document

    def _parse_tokens(self, jtokens, ltokens, rtokens, jpos, dataset):
        doc_tokens = []
        char_vocab = ['<PAD>'] + list(string.printable) + ['<EOT>', '<UNK>']
        # full document encoding including special tokens ([CLS] and [SEP]) and byte-pair encodings of original tokens

        special_tokens_map = self._tokenizer.special_tokens_map
        doc_encoding = [self._tokenizer.convert_tokens_to_ids(special_tokens_map['cls_token'])]
        seg_encoding = [0]
        char_encoding = []

        poss = [self.POS_MAP.index(pos) if pos in self.POS_MAP else self.POS_MAP.index("<UNK>") for pos in jpos]

        # parse tokens
        for token_phrase in ltokens:
            token_encoding = self._tokenizer.encode(token_phrase, add_special_tokens=False)
            doc_encoding += token_encoding
            seg_encoding += [0] * len(token_encoding)
        
        for i, token_phrase in enumerate(jtokens):
            
            # if random.random() < 0.12:
            #     token_phrase = "[MASK]"
            # if self.build_vocab and token_phrase.lower() not in self.word2inx:
            #     self.word2inx[token_phrase.lower()] = len(self.word2inx)
            token_encoding = self._tokenizer.encode(token_phrase, add_special_tokens=False)
            # token_encoding_char = list(char_vocab.index(c) for c in token_phrase)
            token_encoding_char = []
            for c in token_phrase:
                if c in char_vocab:
                    token_encoding_char.append(char_vocab.index(c))
                else:
                    token_encoding_char.append(char_vocab.index("<UNK>"))
            span_start, span_end = (len(doc_encoding), len(doc_encoding) + len(token_encoding))
            char_start, char_end = (len(char_encoding), len(char_encoding) + len(token_encoding_char))
            # try:
            if token_phrase.lower() in  self.word2inx:
                inx = self.word2inx[token_phrase.lower()]
            else:
                inx = self.word2inx["<unk>"]
            token = dataset.create_token(i, span_start, span_end, token_phrase, poss[i], inx, char_start, char_end)
            doc_tokens.append(token)
            doc_encoding += token_encoding
            seg_encoding += [1] * len(token_encoding)
            token_encoding_char += [char_vocab.index('<EOT>')]
            char_encoding.append(token_encoding_char)
            # except:
            #     print(jtokens)
        
        for token_phrase in rtokens:
            token_encoding = self._tokenizer.encode(token_phrase, add_special_tokens=False)
            doc_encoding += token_encoding
            seg_encoding += [0] * len(token_encoding)

        doc_encoding += [self._tokenizer.convert_tokens_to_ids(special_tokens_map['sep_token'])]
        seg_encoding += [0]

        return doc_tokens, doc_encoding, char_encoding, seg_encoding

    def _parse_entities(self, jentities, doc_tokens, dataset, mapping_region) -> List[Entity]:
        entities = []

        for entity_idx, jentity in enumerate(jentities):
            entity_type = self._entity_types[jentity['type']]
            start, end = jentity['start'], jentity['end']

            # create entity mention
            tokens = doc_tokens[start:end]
            phrase = " ".join([t.phrase for t in tokens])
            entity = dataset.create_entity(entity_type, tokens, phrase, mapping_region)
            entities.append(entity)

        return entities

    def _parse_regions(self, can_num, iou_value=0.5, normalize=False):
        xmls = os.listdir(self.xml_path)
        res_dict = {}
        for xml in tqdm(xmls, desc="Parsing images."):
            img_id = xml.split('.')[0]
            tree = ET.parse(os.path.join(self.xml_path, xml))
            root = tree.getroot()
            res_dict[img_id] = {"bbox": [], "aspect": [], "box_features": []}
            aspects = []
            gt_boxes = []
            for object_container in root.findall('object'):
                for names in object_container.findall('name'):
                    box_name = names.text
                    box_container = object_container.findall('bndbox')
                    if len(box_container) > 0:
                        xmin = int(box_container[0].findall('xmin')[0].text)
                        ymin = int(box_container[0].findall('ymin')[0].text)
                        xmax = int(box_container[0].findall('xmax')[0].text)
                        ymax = int(box_container[0].findall('ymax')[0].text)
                    aspects.append(box_name)
                    gt_boxes.append([xmin, ymin, xmax, ymax])
            assert len(aspects) == len(gt_boxes)
            bounding_boxes = np.zeros((can_num, 4), dtype=np.float32)
            image_feature = np.zeros((can_num, 2048), dtype=np.float32)
            img_path = os.path.join(self.detection_path, img_id + '.jpg.npz')
            crop_img = np.load(img_path)
            image_num = crop_img['num_boxes']
            final_num = min(image_num, can_num)
            bounding_boxes[:final_num] = crop_img['bounding_boxes'][:final_num]
            image_feature_ = crop_img['box_features']
            if normalize:
                image_feature_ = (image_feature_ / np.sqrt((image_feature_ ** 2).sum()))
            image_feature[:final_num] = image_feature_[:final_num]
            for aspect, gt_box in zip(aspects, gt_boxes):
                IoUs = (torchvision.ops.box_iou(torch.tensor([gt_box]), torch.tensor(bounding_boxes))).numpy()  # (1,x)
                IoU = IoUs[0]
                for i, iou in enumerate(IoU):
                    if len(res_dict[img_id]["bbox"]) == 0 or not np.any(
                            np.all(bounding_boxes[i] == res_dict[img_id]["bbox"], axis=1)):
                        res_dict[img_id]["bbox"].append(bounding_boxes[i].tolist())
                        res_dict[img_id]["box_features"].append(image_feature[i].tolist())
                        res_dict[img_id]["aspect"].append("N")
                    if iou >= iou_value:
                        res_dict[img_id]["aspect"][i] = aspect
            assert len(res_dict[img_id]["bbox"]) == len(res_dict[img_id]["aspect"])
        imags_list = [x.split('.')[0] for x in os.listdir(self.detection_path)]
        xml_img_list = [x.split('.')[0] for x in xmls]
        for img_id in tqdm(imags_list, desc="Parsing image..."):
            if img_id not in xml_img_list:
                res_dict[img_id] = {"bbox": [], "aspect": [], "box_features": []}
                bounding_boxes = np.zeros((can_num, 4), dtype=np.float32)
                image_feature = np.zeros((can_num, 2048), dtype=np.float32)
                img_path = os.path.join(self.detection_path, img_id + '.jpg.npz')
                crop_img = np.load(img_path)
                image_num = crop_img['num_boxes']
                final_num = min(image_num, can_num)
                bounding_boxes[:final_num] = crop_img['bounding_boxes'][:final_num]
                image_feature_ = crop_img['box_features']
                if normalize:
                    image_feature_ = (image_feature_ / np.sqrt((image_feature_ ** 2).sum()))
                image_feature[:final_num] = image_feature_[:final_num]
                for i in range(final_num):
                    res_dict[img_id]["bbox"].append(bounding_boxes[i].tolist())
                    res_dict[img_id]["box_features"].append(image_feature[i].tolist())
                    res_dict[img_id]["aspect"].append("N")
            assert len(res_dict[img_id]["bbox"]) == len(res_dict[img_id]["aspect"])
        return res_dict

    def _parse_relations(self, jrelations, entities, dataset) -> List[Relation]:
        relations = []

        for jrelation in jrelations:
            relation_type = self._relation_types[jrelation['type']]

            head_idx = jrelation['head']
            tail_idx = jrelation['tail']

            # create relation
            head = entities[head_idx]
            tail = entities[tail_idx]

            reverse = int(tail.tokens[0].index) < int(head.tokens[0].index)

            # for symmetric relations: head occurs before tail in sentence
            if relation_type.symmetric and reverse:
                head, tail = util.swap(head, tail)

            relation = dataset.create_relation(relation_type, head_entity=head, tail_entity=tail, reverse=reverse)
            relations.append(relation)

        return relations
