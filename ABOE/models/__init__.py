# -*- coding: utf-8 -*-


from models.towe_first_last_meanpool import TOWE_First_Last_MeanPool

MODEL_CLASS_MAP = {
    'towe_first_last_meanpool_bert':TOWE_First_Last_MeanPool,
    'towe_first_last_meanpool_roberta':TOWE_First_Last_MeanPool,
}

INPUT_FIELDS_MAP = {
    'towe_first_last_meanpool_bert': ['text_token_indices', 'text_tokens_mask', 'target_tag_list', 'target_tag_mask', 'label_tag_list',  'label_tag_mask','target_boundary_indices'],
    'towe_first_last_meanpool_roberta': ['text_token_indices', 'text_tokens_mask', 'target_tag_list', 'target_tag_mask', 'label_tag_list',  'label_tag_mask','target_boundary_indices'],
}

def get_model(model_name):
    return MODEL_CLASS_MAP[model_name], INPUT_FIELDS_MAP[model_name]
