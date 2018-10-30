from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge

gts = {
    184321: [' This is an image of a beach on a cloudy day'],
    # 81922: ['a large jetliner flying over a traffic filled street']
}
res = {
    184321: [' There are  kits flying in the air at the beach'],
    # 81922: ['plane is flying through the sky']
}

def bleu():
    scorer = Bleu(n=4)
    # scorer += (hypo[0], ref1)   # hypo[0] = 'word1 word2 word3 ...'
    #                                 # ref = ['word1 word2 word3 ...', 'word1 word2 word3 ...']
    score, scores = scorer.compute_score(gts, res)

    print('belu = %s' % scores)

def cider():
    scorer = Cider()
    # scorer += (hypo[0], ref1)
    (score, scores) = scorer.compute_score(gts, res)
    print('cider = %s' % score)

def meteor():
    scorer = Meteor()
    score, scores = scorer.compute_score(gts, res)
    print('meter = %s' % score)

def rouge():
    scorer = Rouge()
    score, scores = scorer.compute_score(gts, res)
    print('rouge = %s' % score)

def main():
    # bleu()
    # cider()
    meteor()
    # rouge()
main()