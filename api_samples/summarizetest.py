from aylienapiclient import textapi
client = textapi.Client("2cd4ceb2","7e03cb561407a5ca2ac8dd11001faed9")
file = open("text","r")
text = file.read()
print(len(text))
print('---------------------------------------------')
s = client.Summarize({'title':'bees','text':text})
length = 0
for line in s['sentences']:
    print(line)
    length+=len(line)
print(length)