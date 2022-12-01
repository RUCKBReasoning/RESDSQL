from sklearn.metrics import classification_report

def lista_equal_to_listb(lista, listb):
    return sorted(lista) == sorted(listb)

def cls_metric(ground_truth_labels, predict_labels):
    cls_report = classification_report(
        y_true = ground_truth_labels, 
        y_pred = predict_labels, 
        target_names = ["negatives", "positives"], 
        digits = 4, 
        output_dict = True
    )
    
    return cls_report

def table_ranking_metric(
    pred_table_ranking_indices, 
    gt_table_labels_for_table_ranking,
):
    preds = []

    for data_id in range(len(pred_table_ranking_indices)):
        used_table_ids = []
        for label_id, label in enumerate(gt_table_labels_for_table_ranking[data_id]):
            if label == 1:
                used_table_ids.append(label_id)
        
        used_table_num = len(used_table_ids)
        predicted_ranking_ids = pred_table_ranking_indices[data_id][: used_table_num]

        if lista_equal_to_listb(predicted_ranking_ids, used_table_ids):
            preds.append(1)
        else:
            preds.append(0)
    
    table_ranking_score = sum(preds) / len(preds)
    
    return table_ranking_score


def column_ranking_metric(
    pred_column_ranking_indices, 
    gt_column_labels_for_column_ranking
):
    preds = []

    for data_id in range(len(pred_column_ranking_indices)):
        pred_column_ranking_indices_in_one_db = pred_column_ranking_indices[data_id]
        gt_column_labels_in_one_db = gt_column_labels_for_column_ranking[data_id]

        for table_id in range(len(pred_column_ranking_indices_in_one_db)):
            used_column_ids = []
            for label_id, label in enumerate(gt_column_labels_in_one_db[table_id]):
                if label == 1:
                    used_column_ids.append(label_id)
            
            used_column_num = len(used_column_ids)

            if used_column_num == 0:
                continue

            predict_ranking_ids = pred_column_ranking_indices_in_one_db[table_id][: used_column_num]

            if lista_equal_to_listb(predict_ranking_ids, used_column_ids):
                preds.append(1)
            else:
                preds.append(0)
        
    column_ranking_score = sum(preds)/len(preds)
    
    return column_ranking_score