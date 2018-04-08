from django.shortcuts import render
from django.http import HttpRequest
test_var = "wejoifwejoifewj"
def entry_page(request):
    return render(request, 'entryPage.html')
"""def summary_quiz_page(request):
    if request.method == "POST":
        var_dict = dict(request.POST)
        if "youtube_link" in var_dict:
            youtube_url = var_dict["youtube_link"][0]
            youtube_id = youtube_url[youtube_url.find("?v=")+3:]
        elif "textarea" in var_dict:
            text_area = var_dict["textarea"][0]
    return render(request, 'summaryQuizPageText.html')
def summary_quiz_page2(request):
    return render(request, 'summaryQuizPage.html')


    Formatting of requests:
    gap: [{"original setnence": ... , "gap-sentence":..., "answer":<index of answer>, "distractors":[d1, d2, d3, d4]} ...]
    summary [{sentence:..., time:int seconds} } ... ]
    captions = [[sentence, time]...]
    """
gap = [{"original_sentence":"This is the original sentence", "gap_sentence":"This is the gap sentence", "answer":3, "distractors":["a","b","c", "d"]},
{"original_sentence":"This is the original sentence2", "gap_sentence":"This is the gap sentence2", "answer":1, "distractors":["e","f","g", "h"]}
]

summary = [{"sentence":"sentence1", "time":1}, {"sentence":"sentence2", "time":2},{"sentence":"sentence3", "time":3},{"sentence":"sentence4", "time":4}, ]
captions =[["sentence1", 1], ["sentence2", 2], ["senence3", 3]]
def link_entered_page(request):
    if request.method == "POST":
        var_dict = dict(request.POST)
        print(var_dict)
        youtube_url = var_dict["youtube_link"][0]
        youtube_id = youtube_url[youtube_url.find("?v=")+3:]
        print(youtube_id)
    global gap
    global summary
    global captions
    response_dict = {"gap":gap, "summary":summary, "captions":captions}
    return render(request, 'summaryQuizPage.html', response_dict)
def text_entered_page(request):
    if request.method == "POST":
        var_dict = dict(request.POST)
        print(var_dict)
        text_area = var_dict["text_area"][0]
        print(text_area)
    return render(request, 'summaryQuizPageText.html', {"content":["hello","hello"] })
