from pycocoevalcap.bleu.bleu_scorer import BleuScorer
from pycocoevalcap.cider.cider_scorer import CiderScorer
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge

with open("examples/hypo.txt") as f:
    hyp = f.readlines()
    hypo = [x.strip() for x in hyp]
print(hypo)
with open("examples/ref1.txt") as f:
    ref = f.readlines()
    ref1 = [x.strip() for x in ref]
print(ref1)
with open("examples/ref2.txt") as f:
    ref = f.readlines()
    ref2 = [x.strip() for x in ref]
print(ref2)

hypo = ['this is the model generated sentence1 which seems good enough']
ref1 = ['this is the model generated sentence1 which seems good enough']

def bleu():
    scorer = BleuScorer(n=4)
    scorer += (hypo[0], ref1)   # hypo[0] = 'word1 word2 word3 ...'
                                    # ref = ['word1 word2 word3 ...', 'word1 word2 word3 ...']
    score, _ = scorer.compute_score()
    print(score)

def cider():
    scorer = CiderScorer()
    scorer += (hypo[0], ref1)
    (score, scores) = scorer.compute_score()
    print(score)

def meteor():
    scorer = Meteor()
    score = scorer._score(hypo[0], ref1)
    print(score)

def rouge():
    scorer = Rouge()
    score = scorer.calc_score(hypo, ref1)
    print(score)

def main():
    # bleu()
    cider()
    # meteor()
    # rouge()
main()

