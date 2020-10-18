import praw
import numpy as np
import pandas as pd

reddit = praw.Reddit(client_id='C4HotN7HiKk-JQ',
                     client_secret='Z7vCqdCnPiucthoPAuXavsQ0fXk',
                     username='ARNisUsername',
                     password='*******',
                     user_agent='isBasedBot')


subreddit = reddit.subreddit('politicalcompassmemes')

Top = subreddit.top(limit=250)

allBased = []
notBased = []
counter = 1
for submission in Top:
    if not submission.stickied:
        submission.comments.replace_more(limit=0)
        comments = submission.comments
        for comment in comments:
            if len(comment.replies) > 0:
                prevReply = comment.body
                for reply in comment.replies:
                    replybody = reply.body
                    if 'based' in replybody or 'Based' in replybody:
                        if 'bot' not in replybody and int(prevReply.count('\n')) < 4 and 'based' not in prevReply and 'Based' not in prevReply:
                            allBased.append(prevReply)
                    prevReply = reply.body
        print(f'{counter}/250 submissions reviewed!')
        counter += 1

allBased = np.array(allBased)
based250 = np.array(['based' for i in range(len(allBased))])

df = pd.DataFrame({
    'message':allBased,
    'is_based':based250
})

df.to_csv('checkBased.csv',index=False)


