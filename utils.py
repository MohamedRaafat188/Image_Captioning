import nltk
import json
from pycocotools.coco import COCO


def clean_sentences(idx2word, generated_lists):
    if type(generated_lists[0]) == list:
        generated_lists = [[lst[i] for lst in generated_lists if lst[i] not in (0, 1, 18)] for i in range(len(generated_lists[0]))]
        sentences = [[idx2word[id] for id in generated_lists[i]] for i in range(len(generated_lists))]
        sentences = [' '.join(lst) for lst in sentences]
        return sentences

    elif type(generated_lists[0]) == int:
        sentence = [idx2word[id] for id in generated_lists if id not in (0, 1, 18)]
        return sentence


def calculate_bleu_scores(ref_coco, predictions_file):
    hypotheses = []
    referances = []
    predictions = json.load(open(predictions_file, 'r'))

    weights = [(1, 0, 0, 0), (0.5, 0.5, 0, 0), (1/3, 1/3, 1/3, 0), (0.25, 0.25, 0.25, 0.25)]
    smooth_func = nltk.translate.bleu_score.SmoothingFunction().method1      # type: ignore

    for item in predictions:
        prediction = item["caption"].rstrip(' .')
        prediction = nltk.tokenize.word_tokenize(prediction.lower())
        hypotheses.append(prediction)

        gt_captions = [ann["caption"].rstrip('. ') for ann in ref_coco.imgToAnns[item["image_id"]]]
        gt_captions = [nltk.tokenize.word_tokenize(caption.lower()) for caption in gt_captions]
        referances.append(gt_captions)

    bleus = nltk.translate.bleu_score.corpus_bleu(referances, hypotheses, weights, smooth_func)     # type: ignore        
    
    return bleus
    