from django.shortcuts import render
from django.http import HttpRequest
def entry_page(request):
    return render(request, 'entryPage.html')
def summary_quiz_page(request):
    if request.method == "POST":
        print(request.POST)
    return render(request, 'summaryQuizPageText.html')
