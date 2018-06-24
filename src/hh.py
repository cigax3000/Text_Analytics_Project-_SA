my_list = [('this', 'p'), ('product', 'n'), ('is', 'v'), ('funny', 'a'), ('and', 'p'), \
('scary', 'a'), ('and', 'p'), ('it', 'p'), ('is', 'v'), ('an', 'p'), ('insanely', 'a'), \
('great', 'a'), ('movie', 'n'), ('loved', 'v'), ('the', 'p'), \
('exciting', 'a'), ('ending', 'n')]

previous_adjective = False
my_important_terms = []
my_important_term = []

for (w, pos_tag) in my_list:
    if not previous_adjective:
        if my_important_term:
            my_important_terms.append(my_important_term)
        my_important_term = []
    if pos_tag.startswith('n'):
        my_important_term.append(w)
        previous_adjective = False
    if pos_tag.startswith('a'):
        my_important_term.append(w)
        previous_adjective = True
    else:
        previous_adjective = False

if my_important_term:
    my_important_terms.append(my_important_term)


my_important_terms = ["_".join([w for w in important_term]) \
for important_term in my_important_terms]

print(' '.join([term for term in my_important_terms]))