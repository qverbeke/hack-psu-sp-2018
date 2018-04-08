from django.shortcuts import render
from django.http import HttpRequest
def entry_page(request):
    return render(request, 'entryPage.html')
def summary_quiz_page(request):
    if request.method == "POST":
        print(request.POST)
    string = ""
    youtube_id = string[string.find("?v=")+3:]
    return render(request, 'summaryQuizPageText.html')
def summary_quiz_page2(request):
    return render(request, 'summaryQuizPage.html')
