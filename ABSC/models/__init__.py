# -*- coding: utf-8 -*-


from models.roberta_first_last_meanpool import Roberta_First_Last_MeanPool


MODEL_CLASS_MAP = {
    'roberta_first_last_meanpool': Roberta_First_Last_MeanPool,
    'bert_first_last_meanpool': Roberta_First_Last_MeanPool,
}

INPUT_FIELDS_MAP = {
    'roberta_first_last_meanpool': ['text_indices', 'text_mask', 'aspect_boundary_indices', 'aspect_indices', 'aspect_mask'],
    'bert_first_last_meanpool': ['text_indices', 'text_mask', 'aspect_boundary_indices', 'aspect_indices', 'aspect_mask'],
}

def get_model(model_name):
    return MODEL_CLASS_MAP[model_name], INPUT_FIELDS_MAP[model_name]
