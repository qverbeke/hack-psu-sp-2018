from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types

def getsalience(sentence):
    client = language.LanguageServiceClient()
    # text = 'Arjun ran to the store to get the gucci gang album for Christie in order to prove that Dylan is better than Matt.'
    document = types.Document(
        content=sentence,
        type=enums.Document.Type.PLAIN_TEXT)

    entities = client.analyze_entities(document).entities

    rank = []
    for entity in entities:
        rank.append((entity.name, entity.salience))
    try:
        keywords = [item[0] for item in sorted(rank, key = lambda x: x[1], reverse = True)]
    except:
        keywords = []
    return keywords

# getsalience('John is an entrepreneur,  he writes books, runs DFTBA, Vlogbrothers, and Mental_Floss and creates movies, but he can\'t  do everything he wants to do.')
