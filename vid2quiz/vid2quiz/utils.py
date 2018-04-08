########################################################
#Comment: This is the main file for all the api calls
#functions:
#   get_gap_sentences()
#   get_summary()
########################################################
from .youtubecall import call_youtube
from .summarycall import summarize
from .preprocessing import preprocess
from .gapify import generate_gap_sentences
from difflib import SequenceMatcher
import subprocess

# entity_type = ('UNKNOWN', 'PERSON', 'LOCATION', 'ORGANIZATION',
#                    'EVENT', 'WORK_OF_ART', 'CONSUMER_GOOD', 'OTHER'}
def similar(s1, s2):
    return SequenceMatcher(None, s1, s2).ratio()

class Vid2Quiz(object):
    def __init__(self, video_id):
        self.video_id = video_id
        self.clean_text = ""
        self.video_length = 0
        self.k = 0 #number of cluster
        self.youtube_captions = []
        self.cloud_summary = []
        self.summarizer_raw = {}
        self.gap_questions = []

        #input/ output director for BTM
        self.input_dir = '/home/herobaby71/Vid2Quiz/BTM/sample-data/'
        self.input_file = 'doc_info.txt'

    def get_youtube_captions(self):
        return self.youtube_captions

    def get_youtube_full_text(self):
        return self.clean_text

    def get_cloud_summary(self):
        return self.cloud_summary

    def _get_youtube_transcription(self):
        """
            getting youtube captions and save it
        """
        youtube_captions = call_youtube(self.video_id)
        youtube_cap_lst = youtube_captions.split('\n\n')
        youtube_cap_lst = [item.split('\n') for item in youtube_cap_lst]

        corpus = ""
        total_time = 0
        new_yt_captions = []
        for i, row in enumerate(youtube_cap_lst):
            if(len(row) > 2):
                #get the seconds
                timedelta = [item.strip() for item in row[1].split('-->')]
                start_time = timedelta[0].split(',')[0]
                h, m, s = start_time.split(':')
                secs = int(h) * 3600 + int(m) * 60 + int(s)
                total_time=secs

                #get the corpus
                sentence = ''
                for j in range(2, len(row)):
                    sentence = ' '.join((sentence, row[j]))
                new_yt_captions.append([secs, sentence])
                corpus = ' '.join((corpus, sentence))
        self.video_length = total_time
        self.k = int(self.video_length % 65)
        self.youtube_captions = new_yt_captions
        self.clean_text = corpus

    def _get_summary_from_text(self):
        """
            get the summary from the text
        """
        summarizer_raw = summarize(self.clean_text)
        cloud_summary = []
        for i, sentence in enumerate(summarizer_raw['sentences']):
            for j,caption in enumerate(self.youtube_captions):
                if(similar(caption[1], sentence) > .5):
                    cloud_summary.append({'sentence':sentence,'time':caption[0]})
                    break
        self.cloud_summary = cloud_summary

    def _preprocess_gapify_BTM(self):
        """
            runBTM params:
                $0: K,
                $1: input_dir,
                $2: output_dir,
                $3: doc_file
        """
        corpus_path = self.input_dir + self.input_file
        preprocess(self.clean_text, outpath=corpus_path)

        # subprocess.check_call(['/home/herobaby71/Vid2Quiz/BTM/script/runBTM.sh', self.k, self.input_dir, self.output_dir, self.doc_file])
        # subprocess.Popen(['/home/herobaby71/Vid2Quiz/BTM/script/runBTM.sh %s %s %s %s' %(self.k,self.input_dir,self.output_dir,self.doc_file)], shell = True)
        subprocess.call('/home/herobaby71/Vid2Quiz/BTM/script/runBTM.sh', shell = True)

        #getting gap sentences
        self.gap_questions = generate_gap_sentences(self.clean_text, topn = 10)

    def get_gap_sentences(self):
        """
            make gap out of the sentence
        """
    #
# vid2quiz = Vid2Quiz("3ez10ADR_gM")
