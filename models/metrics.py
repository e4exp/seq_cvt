from logging import getLogger
logger = getLogger(__name__)

from numpy.core.fromnumeric import mean, std


def error_exact(htmls_gt, htmls_pred):
    errors = []

    # for test files
    logger.debug(len(htmls_gt))
    logger.debug(len(htmls_pred))

    for i in range(len(htmls_gt[0])):
        html_pred = htmls_pred[i]
        html_gt = htmls_gt[i][0]

        # for html tags
        length = max(len(html_pred), len(html_gt))
        if len(html_pred) == length:
            # pad html_gt
            html_gt = html_gt + ["無効"] * (length - len(html_gt))
        else:
            #pad html_pred
            html_pred = html_pred + ["無効"] * (length - len(html_pred))

        error = 0
        for j, tag in enumerate(html_pred):
            if tag != html_gt[j]:
                error += 1
        error /= length
        errors.append(error)
    error_mean = mean(errors)
    error_std = std(errors)
    return error_mean, error_std


def accuracy_exact(htmls_gt, htmls_pred):
    scores = []

    # for test files
    logger.debug(len(htmls_gt))
    logger.debug(len(htmls_pred))

    for i in range(len(htmls_gt[0])):
        html_pred = htmls_pred[i]
        html_gt = htmls_gt[i][0]

        # for html tags
        length = max(len(html_pred), len(html_gt))
        len_stop = min(len(html_pred), len(html_gt))
        score = 0
        for j, tag in enumerate(html_pred):
            if j >= len_stop:
                break
            if tag == html_gt[j]:
                score += 1
        score /= length
        scores.append(score)

    logger.debug(scores)
    score_mean = mean(scores)
    score_std = std(scores)
    return score_mean, score_std
