from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types

# Instantiates a client
client = language.LanguageServiceClient()
text = 'Arjun ran to the store to get the gucci gang album for Christie in order to prove that Dylan is better than Matt.'
document = types.Document(
    content=text,
    type=enums.Document.Type.PLAIN_TEXT)
entities = client.analyze_entities(document).entities
print(entities)
for entity in entities:
    print(entity.name,entity.salience)
