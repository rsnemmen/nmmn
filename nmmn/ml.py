"""
Machine learning methods
=========================

"""

import numpy as np
#from . import misc





def AUCmulti(y_true, y_score):
    """
Computes the area under the ROC curve for multiclass classification models. 
Useful for evaluating the performance of such a model.

Assume `y_true` contains the true labels and `y_score` contains predicted probabilities 
for each class.

:param y_true: 1D array listing the labels
:param y_score: multidimensional array of predicted probabilities

Example: AUC for a classification involving 7 labels and 10 instances.

    # Mock data
    ytrue=np.array([6, 2, 6, 6, 6, 6, 5, 1, 5, 0])
    y_score=np.array([[0.11, 0.04, 0.  , 0.  , 0.03, 0.12, 0.69],
       [0.  , 0.03, 0.76, 0.  , 0.  , 0.01, 0.13],
       [0.05, 0.01, 0.  , 0.  , 0.  , 0.27, 0.63],
       [0.09, 0.01, 0.  , 0.  , 0.  , 0.47, 0.43],
       [0.09, 0.  , 0.01, 0.  , 0.08, 0.51, 0.31],
       [0.03, 0.53, 0.  , 0.  , 0.03, 0.17, 0.21],
       [0.17, 0.07, 0.01, 0.  , 0.03, 0.36, 0.32],
       [0.08, 0.3 , 0.09, 0.  , 0.05, 0.16, 0.26],
       [0.01, 0.01, 0.  , 0.  , 0.01, 0.6 , 0.33],
       [0.  , 0.04, 0.08, 0.01, 0.  , 0.37, 0.41]])

    AUCmulti(ytrue, yscore)
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import label_binarize

    # Binarize the labels for a multi-class problem
    y_true = label_binarize(y_true, classes=range(y_score.shape[1]))

    # Compute the AUC for each class
    auc = roc_auc_score(y_true, y_score, multi_class='ovr')

    return auc



def show_token_pos_ids(model, text: str):
    """
    Displays a formatted table showing token positions, IDs, and string representations.
    Useful for debugging and understanding how a model tokenizes input text.

    Assumes `model` has methods `to_tokens()`, `to_str_tokens()` for tokenization.
    Model example: `model = HookedTransformer.from_pretrained(model_name, device=device)` from
    `transformer_lens`.

    :param model: A tokenizer model with `to_tokens()` and `to_str_tokens()` methods
    :param text: The input string to tokenize and display

    :return: A tuple containing (tokens, token_ids, str_tokens, positions)
             - tokens: 2D tensor of shape [1, seq_len]
             - token_ids: List of integer token IDs
             - str_tokens: List of string representations of tokens
             - positions: List of positional indices (0 to seq_len-1)

    Example: Visualizing tokenization for a simple sentence.

    >>> text = "The quick brown fox jumps over the lazy dog."
    >>> tokens_cpu, token_ids, str_tokens, positions = show_token_pos_ids(model, text)
      0        1    2      3      4     5      6    7     8    9   10
    50256     464  2068   7586  21831 18045   625  262  16931 3290 13
<|endoftext|> The  quick  brown  fox   jumps  over  the  lazy  dog . 
    """
    tokens = model.to_tokens(text)               # [1, seq_len]
    token_ids = tokens[0].tolist()               # list[int]
    str_tokens = model.to_str_tokens(tokens[0])  # list[str]
    positions = list(range(len(token_ids)))      # 0..seq_len-1

    # Column width: big enough for position, id, or token string
    widths = [
        max(len(str(p)), len(str(tid)), len(tok))
        for p, tid, tok in zip(positions, token_ids, str_tokens)
    ]

    pos_line = " ".join(f"{p:^{w}}" for p, w in zip(positions, widths))
    id_line  = " ".join(f"{tid:^{w}}" for tid, w in zip(token_ids, widths))
    tok_line = " ".join(f"{tok:^{w}}" for tok, w in zip(str_tokens, widths))

    print(pos_line)
    print(id_line)
    print(tok_line)

    return tokens, token_ids, str_tokens, positions