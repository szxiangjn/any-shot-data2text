import re
from nltk.tokenize import word_tokenize
from table_text_eval import parent

def parent_score(preds, refs, triples):
    # convert preds
    preds = [word_tokenize(pred.lower()) for pred in preds]

    # convert refs
    refs = [[word_tokenize(r.lower()) for r in ref] for ref in refs]

    # convert triples
    tables = []
    for triple in triples:
        ts = triple.split("<H> ")[1:]
        table = []
        for t in ts:
            t = t.strip()
            t = t.replace("<R>", "|").replace("<T>", "|").lower()
            t = t.split(" | ")
            table.append(tuple([item.split() for item in t]))
        tables.append(table)
    
    precision, recall, f1, _ = parent(preds, refs, tables)
    return precision, recall, f1