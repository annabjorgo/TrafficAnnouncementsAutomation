import pyspark.sql.functions as sf


def measure_accuracy(aligned_list, pd_link, df):
    def correct_by_recordId(input_list):
        correct_recordId_list = []
        for prediction in input_list:
            nrk_id = prediction['nrk_id']
            corr_recordId = str(pd_link[pd_link['nrk_id'] == nrk_id].iloc[0]['svv_id']).strip()
            pred_recordId = str(prediction['recordId']).strip()

            if corr_recordId == pred_recordId:
                correct_recordId_list.append(prediction)
        return correct_recordId_list

    def correct_by_situationId(input_list):
        correct_situationId_list = []
        for prediction in input_list:
            nrk_id = prediction['nrk_id']
            corr_situationId = str(pd_link[pd_link['nrk_id'] == nrk_id].iloc[0]['situationId']).strip()
            pred_situationId = str(prediction['situationId']).strip()

            if corr_situationId == pred_situationId:
                correct_situationId_list.append(prediction)
        return correct_situationId_list

    def correct_by_location(input_list):
        group_df = df.groupby(sf.year("overallStartTime"), sf.month("overallStartTime"), sf.day("overallStartTime"),
                              sf.col("locations.coordinatesForDisplay")).agg(
            sf.collect_set(sf.col("recordId")).alias("ids"))
        group_df.persist()

        correct_location_list = []
        for prediction in input_list:
            nrk_id = prediction['nrk_id']
            corr_recordId = str(pd_link[pd_link['nrk_id'] == nrk_id].iloc[0]['svv_id']).strip()
            pred_recordId = str(prediction['recordId']).strip()

            if not group_df.where(
                    sf.array_contains("ids", corr_recordId) & sf.array_contains("ids", pred_recordId)).isEmpty():
                correct_location_list.append(prediction)
        group_df.unpersist()
        return correct_location_list

    situation_id_list = correct_by_situationId(aligned_list)
    record_id_list = correct_by_recordId(aligned_list)
    location_list = correct_by_location(aligned_list)

    correct_list = list({v['nrk_id']: v for v in (location_list + record_id_list + situation_id_list)}.values())
    incorrect_list = [item for item in aligned_list if item not in correct_list]
    avg_correct_similarity = sum(it['similarity'] for it in correct_list) / len(correct_list)

    print(f"Accuracy for situationId: {len(situation_id_list) / len(aligned_list)}")
    print(f"Accuracy for recordId: {len(record_id_list) / len(aligned_list)}")
    print(f"Accuracy for location: {len(location_list) / len(aligned_list)}")
    print(f"Avg similarity: {avg_correct_similarity}")

    return correct_list, incorrect_list


def check_incorrect(incorrect_list, pd_link, pd_df):
    analysed_list = []
    for i, it in enumerate(incorrect_list):
        try:
            corr_nrk_id = it['nrk_id']
            corr_svv_id = pd_link[pd_link['nrk_id'] == corr_nrk_id].iloc[0]['svv_id']
            pred_svv_id = it['recordId']

            nrk_corr_text = pd_link[pd_link['nrk_id'] == corr_nrk_id].iloc[0]['nrk_text']
            # fixme: this only takes the first record with the svv_id

            svv_corr_text = pd_df[pd_df['recordId'] == corr_svv_id].iloc[0]['concat_text']
            svv_pred_text = pd_df[pd_df['recordId'] == pred_svv_id].iloc[0]['concat_text']

            analysed_list.append(
                {"correct_nrk_id_text": (corr_nrk_id, nrk_corr_text),
                 "correct_svv_id_text": (corr_svv_id, svv_corr_text),
                 "predicted_svv_id_text": (pred_svv_id, svv_pred_text), "similarity": it['similarity']})
        except:
            pass
    return analysed_list


def print_incorrect(incorrect):
    for it in incorrect:
        print(
            f'Correct nrk id and text: {it["correct_nrk_id_text"][0].strip()}, {it["correct_nrk_id_text"][1].strip()}')
        print(f'Correct svv id and text: {it["correct_svv_id_text"][0]}, {it["correct_svv_id_text"][1]}')
        print(f'Predicted svv id and text: {it["predicted_svv_id_text"][0]}, {it["predicted_svv_id_text"][1]}')
        print(f'Similarity: {it["similarity"]}')
        print("\n")
