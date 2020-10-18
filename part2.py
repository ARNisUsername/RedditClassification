import praw
import numpy as np
import pandas as pd

reddit = praw.Reddit(client_id='C4HotN7HiKk-JQ',
                     client_secret='Z7vCqdCnPiucthoPAuXavsQ0fXk',
                     username='ARNisUsername',
                     password='zero77gold',
                     user_agent='isBasedBot')


subreddit = reddit.subreddit('politicalcompassmemes')

Cont = subreddit.controversial(limit=350)

allBased = []
notBased = []
counter = 1

for submission in Cont:
    if not submission.stickied:
        submission.comments.replace_more(limit=0)
        comments = submission.comments
        for comment in comments:
            theComment = comment.body.rstrip('\\n')
            theComment = theComment.encode('unicode-escape').decode('utf-8')
            theScore = comment.score
            if theScore < -2 and theComment.count('\n') < 4:
                if 'Based' not in theComment and 'based' not in theComment:
                    notBased.append(theComment)
        print(f'{counter}/350 submissions reviewed!')
        counter += 1
        

notBased = np.array(notBased)
notbased350 = np.array(['not based' for i in range(len(notBased))])

otherDf = pd.DataFrame({
    'message':notBased,
    'is_based':notbased350
})

df.to_csv('updatedBased2.csv',index=False)




