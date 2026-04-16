

states = ['High', 'Medium', 'Low']

obs = ['long_session', 'scroll', 'purchase']

start_prob = {'High': 0.3, 'Medium': 0.5, 'Low': 0.2}

trans_prob = {
    'High':   {'High': 0.6, 'Medium': 0.3, 'Low': 0.1},
    'Medium': {'High': 0.2, 'Medium': 0.5, 'Low': 0.3},
    'Low':    {'High': 0.1, 'Medium': 0.3, 'Low': 0.6}
}

emit_prob = {
    'High':   {'long_session': 0.5, 'scroll': 0.2, 'purchase': 0.3},
    'Medium': {'long_session': 0.2, 'scroll': 0.4, 'purchase': 0.1},
    'Low':    {'long_session': 0.1, 'scroll': 0.1, 'purchase': 0.05}
}


def forward(obs):
    fwd = [{s: start_prob[s] * emit_prob[s][obs[0]] for s in states}]

    for t in range(1, len(obs)):
        fwd.append({
            curr: sum(fwd[t-1][prev] * trans_prob[prev][curr] for prev in states)
                  * emit_prob[curr][obs[t]]
            for curr in states
        })

    return sum(fwd[-1].values())



def viterbi(obs):
    V = [{}]
    path = {}

    for s in states:
        V[0][s] = start_prob[s] * emit_prob[s][obs[0]]
        path[s] = [s]

    for t in range(1, len(obs)):
        new_path = {}
        V.append({})

        for curr in states:
            prob, state = max(
                (V[t-1][prev] * trans_prob[prev][curr] * emit_prob[curr][obs[t]], prev)
                for prev in states
            )
            V[t][curr] = prob
            new_path[curr] = path[state] + [curr]

        path = new_path

    prob, state = max((V[-1][s], s) for s in states)
    return prob, path[state]


print("Forward Probability:", forward(obs))

prob, best_path = viterbi(obs)
print("Most Likely Engagement Path:", best_path)
