from django.shortcuts import render
from django.http import HttpRequest
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
    return render(request, 'summaryQuizPage.html')"""
def link_entered_page(request):
    if request.method == "POST":
        var_dict = dict(request.POST)
        print(var_dict)
        youtube_url = var_dict["youtube_link"][0]
        youtube_id = youtube_url[youtube_url.find("?v=")+3:]
        print(youtube_id)
    return render(request, 'summaryQuizPageText.html')
def text_entered_page(request):
    if request.method == "POST":
        var_dict = dict(request.POST)
        print(var_dict)
        text_area = var_dict["text_area"][0]
        print(text_area)
    return render(request, 'summaryQuizPage.html')
