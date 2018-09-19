from pycocoevalcap.bleu.bleu_scorer import BleuScorer
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge

with open("examples/hypo.txt") as f:
    hyp = f.readlines()
    hypo = [x.strip() for x in hyp]
with open("examples/ref1.txt") as f:
    ref = f.readlines()
    ref1 = [x.strip() for x in ref]

hypo = ['plane is flying through the sky']
ref1 = ['a large jetliner flying over a traffic filled street']

def bleu():
    scorer = BleuScorer(n=4)
    scorer += (hypo[0], ref1)   # hypo[0] = 'word1 word2 word3 ...'
                                    # ref = ['word1 word2 word3 ...', 'word1 word2 word3 ...']
    score, _ = scorer.compute_score()
    print('belu = %s' % score)

def cider():
    gts = {184321: ['a train traveling down tracks next to lights'],
           81922: ['a large jetliner flying over a traffic filled street']}
    res = {184321: ['train traveling down a track in front of a road'],
           81922: ['plane is flying through the sky']}

    scorer = Cider()
    # scorer += (hypo[0], ref1)
    (score, scores) = scorer.compute_score(gts, res)
    print('cider = %s' % score)

def meteor():
    scorer = Meteor()
    score = scorer._score(hypo[0], ref1)
    print('meter = %s' % score)

def rouge():
    scorer = Rouge()
    score = scorer.calc_score(hypo, ref1)
    print('rouge = %s' % score)

def main():
    bleu()
    cider()
    meteor()
    rouge()
main()

