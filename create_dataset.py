import pandas as pd

data = {
    'text': ['J\'adore ce produit', 'C\'est la pire chose que j\'ai achetée', 'Je suis neutre à ce sujet', 'C\'est incroyable', 'Je n\'aime pas ça'],
    'label': [1, 0, 1, 1, 0]
}

df = pd.DataFrame(data)

df.to_csv('data.csv', index=False)