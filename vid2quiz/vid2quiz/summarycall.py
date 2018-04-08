from aylienapiclient import textapi
client = textapi.Client("2cd4ceb2","7e03cb561407a5ca2ac8dd11001faed9")
# file = open("text","r")
# text = file.read()
# print(len(text))
# print('---------------------------------------------')


def summarize(text, text_title = "This Is The Title"):
    s = client.Summarize({'title':text_title,'text':text})
    return s
