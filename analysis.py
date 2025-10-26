# analysis.py
# -----------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.

######################
# ANALYSIS QUESTIONS #
######################

def question2a():
    """
      Prefer the close exit (+1), risking the cliff (-10).
    """
    answerDiscount = 0.3
    answerNoise = 0.0
    answerLivingReward = 0.0
    return answerDiscount, answerNoise, answerLivingReward

def question2b():
    """
      Prefer the close exit (+1), but avoiding the cliff (-10).
    """
    answerDiscount = 0.3
    answerNoise = 0.2
    answerLivingReward = 0.0
    return answerDiscount, answerNoise, answerLivingReward

def question2c():
    """
      Prefer the distant exit (+10), risking the cliff (-10).
    """
    answerDiscount = 0.9
    answerNoise = 0.0
    answerLivingReward = 0.0
    return answerDiscount, answerNoise, answerLivingReward

def question2d():
    """
      Prefer the distant exit (+10), avoiding the cliff (-10).
    """
    answerDiscount = 0.9
    answerNoise = 0.2
    answerLivingReward = 0.0
    return answerDiscount, answerNoise, answerLivingReward

def question2e():
    """
      Avoid both exits and the cliff (so an episode should never terminate).
    """
    answerDiscount = 0.9
    answerNoise = 0.2
    answerLivingReward = 1.0
    return answerDiscount, answerNoise, answerLivingReward

if __name__ == '__main__':
    print('Answers to analysis questions:')
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print('  Question %s:\t%s' % (q, str(response)))
