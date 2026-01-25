import spacy
import scispacy
import os
import json
import warnings
from spacy.tokens import Doc
from more_itertools import locate

warnings.filterwarnings("ignore", category=FutureWarning)
nlp = spacy.load("en_core_sci_sm")

def custom_tokenizer(text):
    tokens = text.split(" ")
    return Doc(nlp.vocab, tokens)

nlp.tokenizer = custom_tokenizer

class JsonInputAugmenter():
    def __init__(self):
        date = '0217'  # Date or version identifier for the dataset
        basepath = os.path.join('..', 'InputsAndOutputs', 'input')  # Set your base path here
        
        # Paths to input data files
        self.input_dataset_paths = [
            os.path.join(basepath, 'md_train_KG_' + date + '.json'),
            os.path.join(basepath, 'md_test_KG_' + date + '.json'),
            os.path.join(basepath, 'md_KG_all_' + date + '.json')
        ]
        
        # Paths to output data files
        self.output_dataset_paths = [
            os.path.join(basepath, 'md_train_KG_' + date + '_agu.json'),
            os.path.join(basepath, 'md_test_KG_' + date + '_agu.json'),
            os.path.join(basepath, 'md_KG_all_' + date + '_agu.json')
        ]

    def augment_docs_in_datasets(self):
        for ipath, opath  in zip(self.input_dataset_paths, self.output_dataset_paths):
            self._augment_docs(ipath, opath)

    def _augment_docs(self, ipath, opath):
        global tokens_dict
        print(f"\n" + "="*50)
        print(f"[INFO] Augmenting data from: {os.path.basename(ipath)}")
        print("="*50)
        documents = json.load(open(ipath))
        augmented_documents = []
        nmultiroot=0
        total_docs = len(documents)
        
        for idx, document in enumerate(documents):
            if (idx + 1) % 50 == 0:
                print(f"Processing... {idx + 1}/{total_docs} documents", end='\r')
                
            jtokens = document['tokens']
            jrelations = document['relations']
            jentities = document['entities']
            jorig_id = document.get('orig_id', document.get('origId', 'unknown'))
            jtext = document.get('sents', ' '.join(jtokens))
            lower_jtokens = jtokens
            text = ' '.join(lower_jtokens)
            tokens = nlp(text)
            jtags = [token.tag_ for token in tokens]
            jdeps = [token.dep_ for token in tokens]
            vpos = list(locate(jdeps, lambda x: x == 'ROOT'))
            if (len(vpos) != 1):
                flag = 1
                nmultiroot += 1
            else:
                flag = 0
            
            verb_indicator = [0] * len(jdeps)
            for i in vpos:
                verb_indicator[i] = 1
                
            jdep_heads = []
            for i, token in enumerate(tokens):
              if token.head == token:
                 token_idx = 0
              else:
                 token_idx = token.head.i - tokens[0].i + 1

              jdep_heads.append(token_idx)
            
            d = {"tokens": jtokens, "pos_tags": jtags, "dep_label": jdeps, "verb_indicator": verb_indicator, "dep_head": jdep_heads, "entities": jentities, "relations": jrelations, "orig_id": jorig_id, "sents": jtext}
            if (flag==1):
                pass 

            augmented_documents.append(d)
        print(f"\n[INFO] Saving augmented data to: {os.path.basename(opath)}")
        print(f"       Total docs with multiroot issue: {nmultiroot}")
        with open(opath, "w") as ofile:
            json.dump(augmented_documents, ofile)


if __name__ == "__main__":
    augmenter = JsonInputAugmenter()
    augmenter.augment_docs_in_datasets()