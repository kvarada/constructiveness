import pickle
import json, argparse

from datetime import date, timedelta
import requests, time, simplejson
import pandas as pd

ROOT = '/Users/vkolhatk/Data/Constructiveness/'

def json_pprint(json_file):
    '''
    :param json_file:
    :return:
    '''
    with open(json_file, 'r') as f:
        comments = simplejson.load(f)

    print(json.dumps(comments, sort_keys=True, indent=4, separators=(',', ': ')))
    #print(len(data))

class NYTCommentExtractor:
    def __init__(self):
        '''
        '''
        # Put your API key here.
        self.api_key = 'XXX'
        self.path = ROOT
        self.date_json_path = ROOT + 'intermediate_output/json_files/'

        # Yields an iterator which allows to iterate through date.

    # This function draws from http://stackoverflow.com/a/10688060
    def perdelta(self, start, end, delta):
        curr = start
        while curr < end:
            yield curr
            curr += delta

    def scrape_by_date(self, from_date = date(2017, 5, 15), to_date = date(2017, 6, 6)):
        '''
        :return:
        '''

        # Scrape 300 comments per day
        # For each search, loop tries 4 times to get a valid response.
        # If >4 tries, then loops moves on to next day, dumping that day's comments if any are found into a file.
        # Outputs a JSON file for each day.

        for da in self.perdelta(from_date, to_date, timedelta(days=1)):
        #for da in self.perdelta(date(2017, 5, 15), date(2017, 6, 6), timedelta(days=1)):
            comments = []
            print(da)
            skip = False
            gotany = True

            # Collect 25 comments at a time for 12 times (25*12 = 300 comments)
            for i in range(16):
                if not skip:
                    success = False
                    count = 0

                    # Need to include your own API key here
                    url = ('http://api.nytimes.com/svc/community/v3/user-content/' +
                           'by-date.json?api-key=' + self.api_key +'&date=' + str(da) +
                           '&offset=' + str(25 * i) + '&sort=recommended')
                    print('url: ', url)
                    while not success:
                        comments_data = requests.get(url)
                        try:
                            data = simplejson.loads(comments_data.content)
                            success = True  # go to the next offset
                            for d in data['results']['comments']:
                                comments.append(d)
                            time.sleep(2)
                        except:
                            print('error on {}'.format(str(da)))
                            print(url)
                            count += 1
                            if count > 3:
                                success = True
                                # skip to the next day
                                skip = True
                                if i == 0:
                                    # If we didn't get any comments from that day
                                    gotany = False
                            time.sleep(2)

            # Save data pulled into JSON file
            if gotany:
                filestr = self.date_json_path + 'comments {}.json'.format(str(da))
                print('filestr: ', filestr)
                with open(filestr, 'w') as f:
                    simplejson.dump(comments, f)

        allcomments = []
        for d in self.perdelta(date(2013, 1, 1), to_date, timedelta(days=1)):
            # Don't have to worry about failed comment collections thanks to try/except.
            # If we didn't collect the comments for a given day, the file load fails and it moves on.
            try:
                with open(self.date_json_path + '/comments {}.json'.format(str(d))) as f:
                    c = simplejson.load(f)
                    allcomments.extend(c)
            except Exception:
                pass
        # Save JSON file

        # Note: commented out as the file has already been created. Uncomment if need to start over.

        with open (ROOT + 'tmp/comment_data.json', 'w') as f:
            simplejson.dump(allcomments, f)

      # Load JSON file
        with open(ROOT + 'tmp/comment_data.json', 'r') as f:
            comments = simplejson.load(f)

        print(len(comments))

        # Convert data into a dataframe by creating a dataframe out of a list of dictionaries.
        commentsdicts = []
        editor_picks_commentsdict = []

        # Loop through every comment
        for c in comments:
            d = {}
            d['approveDate'] = c['approveDate']
            d['assetID'] = c['assetID']
            d['assetURL'] = c['assetURL']
            d['commentBody'] = c['commentBody'].replace("<br/>", " ")

            # Calculate word count by splitting on spaces. Treating two, three, etc... spaces as single space.
            d['commentWordCount'] = len(
                c['commentBody'].replace("<br/><br/>", " ").replace("    ", " ").replace("   ", " ").replace("  ",
                                                                                                             " ").split(
                    " "))

            # Count number of letters in each word, divide by word count. Treating two, three, etc... spaces as single space.
            d['averageWordLength'] = float(len(
                c['commentBody'].replace("%", "").replace("&", "").replace("!", "").replace("?", "").replace(",",
                                                                                                             "").replace(
                    "'", "").replace(".", "").replace(":", "").replace(";", "").replace("    ", " ").replace("   ",
                                                                                                             " ").replace(
                    "  ", " ").replace(" ", ""))) / d["commentWordCount"]

            d['commentID'] = c['commentID']
            d['commentSequence'] = c['commentSequence']
            d['commentTitle'] = c['commentTitle']
            d['createDate'] = c['createDate']
            d['editorsSelection'] = c['editorsSelection']
            d['lft'] = c['lft']
            d['parentID'] = c['parentID']
            d['recommendationCount'] = c['recommendationCount']
            d['replies'] = c['replies']
            d['replyCount'] = c['replyCount']
            d['rgt'] = c['rgt']
            d['status'] = c['status']
            d['statusID'] = c['statusID']
            d['updateDate'] = c['updateDate']
            d['userDisplayName'] = c['userDisplayName']
            d['userID'] = c['userID']
            d['userLocation'] = c['userLocation']
            d['userTitle'] = c['userTitle']
            d['userURL'] = c['userURL']

            if d['editorsSelection'] == 1:
                editor_picks_commentsdict.append(d)

            commentsdicts.append(d)

        commentsdf = pd.DataFrame(commentsdicts)
        editor_picks_commentsdf = pd.DataFrame(editor_picks_commentsdict)

        editor_picks_commentsdf.to_csv(ROOT + 'data/NYT_comments_csv/NYT_comments_all_' + to_date.__str__() + '_editor_picks.csv')
        commentsdf.to_csv(ROOT + 'data/NYT_comments_csv/NYT_comments_all_' + to_date.__str__() + '.csv')
        print('The output csv is written!')

    def scrape_test(self, output_pickle_file):
        comments = []
        all_data = []
        urls = ([
             'https://www.nytimes.com/2017/05/08/opinion/john-mccain-rex-tillerson-human-rights.html?ref=opinion',
             'https://www.nytimes.com/2017/04/08/opinion/sunday/hillary-clinton-free-to-speak-her-mind.html'
            ])
        #r = requests.get("http://api.nytimes.com/svc/search/v2/articlesearch.json?q=trump+women+accuse&begin_date=20161001&api-key=XXXXX")
        #url = ('http://api.nytimes.com/svc/community/v3/user-content/url.json' +
        #       'url=https://www.nytimes.com/2017/05/08/opinion/john-mccain-rex-tillerson-human-rights.html?ref=opinion' +
        #       '&api-key=' + self.api_key + '&offset=' + str(50))
        #       #'by-date.json?api-key=' + self.api_key + '&date=' + str(da) +

        #print('url: ', url)
        #r = requests.get(url)
        for url in urls:
            query_url = ('http://api.nytimes.com/svc/community/v3/user-content/'+
                  'url.json?url=' + url +
                  '&api-key=' + self.api_key + '&sort=highlights')
            print(query_url)
            comments_data = requests.get(query_url)
            #data = r.json()

            try:
                data = simplejson.loads(comments_data.content)
                for d in data['results']['comments']:
                    comments.append(d)
                time.sleep(5)
            except:
                print('Could not crawl comments')
                #sys.exit(0)

            print('Number of comments: ', len(comments))

            all_data.append(data)

        f = open(self.path + output_pickle_file, 'wb')
        pickle.dump(all_data, f)
        f.close()

    def explore_data(self, input_pickle_file):
        f = open(self.path + input_pickle_file, 'rb')
        data = pickle.load(f)
        f.close()

        print(json.dumps(data, sort_keys=True, indent=4, separators=(',', ': ')))
        #print(data)
        print(len(data))
        #print(len(data["response"]["docs"]))

def get_arguments():
    parser = argparse.ArgumentParser(description='SFU Sentiment Calculator')

    parser.add_argument('--NYT_picks_pickle_file', '-pp', type=str, dest='picks_pickle', action='store',
                        default= ROOT + 'NYT_picks_pickle_file.pkl',
                        help="the NYT picks pickle file")

    parser.add_argument('--NYT_recommended_pickle_file', '-rp', type=str, dest='recommended_pickle', action='store',
                        default='NYT_recommended_pickle_file.pkl',
                        help="the NYT recommended pickle file")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_arguments()
    print(args)
    CE = NYTCommentExtractor()
    CE.scrape_by_date( date(2018, 6, 5), date(2018, 6, 6))

