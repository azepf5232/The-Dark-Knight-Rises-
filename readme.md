# Final Project


``` python
import requests
from bs4 import BeautifulSoup
import pandas as pd
```

``` python
# URL of the website to scrape

dark_knight_link = "https://imsdb.com/scripts/Dark-Knight-Rises,-The.html"
link_request = requests.get(dark_knight_link)
soup = BeautifulSoup(link_request.content, 'html.parser')

dark_knight_script = soup.select('pre')
dark_knight_script = [x.text.strip() for x in dark_knight_script]

entry_info = dark_knight_script[0].split('\n')

entry_info = [x.strip() for x in entry_info if x.strip() != '']


df_entry_info = pd.DataFrame(entry_info)


characters = []
dialogues = []
current_character = None
current_dialogue = []

for row, line in enumerate(entry_info[10:]):
    if line == line.upper() and not line.endswith('.') and not line.endswith(':') and not line.startswith('INT') and not line.startswith('EXT'):
        characters.append(line)
        next_line = entry_info[10:][row + 1]
        dialogues.append(next_line)
        if current_character is not None:
            characters.append(current_character)
            dialogues.append(' '.join(current_dialogue))
        current_character = line
        current_dialogue = []
    else:
        current_dialogue.append(line)

df_dark_knight = pd.DataFrame({'Character': characters, 'Dialogue': dialogues})

df_dark_knight.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|     | Character | Dialogue                                          |
|-----|-----------|---------------------------------------------------|
| 0   | GORDON    | But I knew Harvey Dent. I was...his               |
| 1   | CIA MAN   | Dr. Pavel, I'm CIA.                               |
| 2   | GORDON    | But I knew Harvey Dent. I was...his friend. An... |
| 3   | DRIVER    | He wasn't alone.                                  |
| 4   | CIA MAN   | Dr. Pavel, I'm CIA. Dr. Pavel nods, nervous. C... |

</div>

``` python
## Making a new column with the first 15 characters of the dialogue to identify unique dialogues for each character

df_dark_knight['Dialogue_15'] = df_dark_knight['Dialogue'].str[:15]


new_df = df_dark_knight.drop_duplicates(
    subset=['Character', 'Dialogue_15'],
    keep='last'
)

new_df = new_df.drop(columns=['Dialogue_15'])

new_df.head
```

    <bound method NDFrame.head of        Character                                           Dialogue
    2         GORDON  But I knew Harvey Dent. I was...his friend. An...
    4        CIA MAN  Dr. Pavel, I'm CIA. Dr. Pavel nods, nervous. C...
    6         DRIVER  He wasn't alone. CIA Man, confused, spots the ...
    7      DR. PAVEL                                           (SHAKEN)
    8        CIA MAN                    You don't get to bring friends.
    ...          ...                                                ...
    2710         FOX                                                   
    2712  (CONFUSED)  Check the user ident on the patch... Tech 2 ty...
    2714      TECH 2  Huh. Bruce Wayne. Fox turns away from the roto...
    2715      ALFRED                   Si, Fernet Branca, per cortesia.
    2716      WAITER                                        Lei e solo?

    [1470 rows x 2 columns]>

``` python
## Number of times a character speaks
character_counts = new_df['Character'].value_counts()

top_characters = character_counts.head(5)

top_characters
```

    Character
    WAYNE     192
    BLAKE     132
    BANE       92
    GORDON     83
    ALFRED     74
    Name: count, dtype: int64

``` python
## Visualization of top speaking characters
import matplotlib.pyplot as plt


top_characters.plot(kind='bar')

plt.title("Top 5 Speaking Characters")
plt.xlabel("Character Name")
plt.ylabel("Number of Lines")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
```

![](readme_files/figure-commonmark/cell-6-output-1.png)

``` python
## Sentiment Analysis of top speaking characters

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

vader = SentimentIntensityAnalyzer()

top_5 = top_characters.index.tolist()

df_top_5 = new_df[new_df['Character'].isin(top_5)]

df_top_5['Sentiment'] = df_top_5['Dialogue'].apply(lambda x: vader.polarity_scores(x)['compound']) 

top5_characters_sentiment = df_top_5.groupby('Character')['Sentiment'].mean().reset_index().sort_values(by='Sentiment', ascending=False)

top5_characters_sentiment
```

    C:\Users\azepf\AppData\Local\Temp\ipykernel_25892\1080326244.py:11: SettingWithCopyWarning:


    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead

    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|     | Character | Sentiment |
|-----|-----------|-----------|
| 4   | WAYNE     | 0.028688  |
| 0   | ALFRED    | -0.095935 |
| 2   | BLAKE     | -0.099490 |
| 1   | BANE      | -0.109113 |
| 3   | GORDON    | -0.114398 |

</div>

``` python
## Vizualization of sentiment scores
import matplotlib.pyplot as plt


top5_characters_sentiment.set_index('Character')['Sentiment'].plot(kind='bar')

plt.title("Top 5 Speaking Characters Sentiment Scores")
plt.xlabel("Character Name")
plt.ylabel("Average Sentiment Score")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
```

![](readme_files/figure-commonmark/cell-8-output-1.png)

``` python
## Sentiment score of Batman vs Bane across movie timeline

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

vader = SentimentIntensityAnalyzer()

characters = ['WAYNE', 'BANE']

df_wayne_bane = new_df[new_df['Character'].isin(characters)].copy()

df_wayne_bane['Sentiment'] = df_wayne_bane['Dialogue'].apply(lambda x: vader.polarity_scores(x)['compound'])

df_wayne_bane['Dialogue_Length'] = df_wayne_bane['Dialogue'].apply(len)

wayne = df_wayne_bane[df_wayne_bane['Character'] == 'WAYNE']
bane = df_wayne_bane[df_wayne_bane['Character'] == 'BANE']
```

``` python
## Sentiment score of Batman vs Bane across movie timeline vizulization

plt.figure(figsize=(12, 6))


plt.hist([wayne['Sentiment'], bane['Sentiment']],
         bins=20,
         label=['Bruce Wayne', 'Bane'])

plt.title("Sentiment Distribution of Dialogue")
plt.xlabel("Sentiment Score")
plt.ylabel("Frequency")
plt.legend()

plt.show()
```

![](readme_files/figure-commonmark/cell-10-output-1.png)

``` python
## Topic modeling of the movie script
from joblib import load, dump
import pandas as pd 
import numpy as np  
import lda
import seaborn as sns
import matplotlib.pyplot as plt 
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pprint as pprint 
import requests
from sklearn.feature_extraction.text import CountVectorizer
```

``` python
stopwords = requests.get('https://raw.githubusercontent.com/stopwords-iso/stopwords-en/refs/heads/master/stopwords-en.txt')

stopwords = stopwords.text.split('\n')

stopword_pattern = r'\b(?:{})\b'.format('|'.join(stopwords))
```

``` python
new_df = new_df.dropna(subset=['Dialogue'])  
new_df["Dialogue"] = new_df["Dialogue"].astype(str)
new_df["Dialogue"] = new_df["Dialogue"].str.lower()

new_df["Dialogue_clean"] = new_df["Dialogue"].str.replace(stopword_pattern, '', regex=True)
new_df["Dialogue_clean"] = new_df["Dialogue_clean"].str.replace(r"[\(\).,?!;:'\"-]", "", regex=True)
new_df["Dialogue_clean"] = new_df["Dialogue_clean"].str.replace(r'\s+', ' ', regex=True)

new_df['word_count'] = new_df['Dialogue_clean'].apply(lambda x: len(x.split()))
new_df['word_count'].describe()

vectorizer = CountVectorizer()
x = vectorizer.fit_transform(new_df['Dialogue_clean'])
vocab = vectorizer.get_feature_names_out()
```

``` python
model = lda.LDA(n_topics=5, n_iter=1500, random_state=1)
model.fit(x)
```

    INFO:lda:n_documents: 1470
    INFO:lda:vocab_size: 3434
    INFO:lda:n_words: 11641
    INFO:lda:n_topics: 5
    INFO:lda:n_iter: 1500
    WARNING:lda:all zero row in document-term matrix found
    INFO:lda:<0> log likelihood: -126546
    INFO:lda:<10> log likelihood: -100840
    INFO:lda:<20> log likelihood: -99120
    INFO:lda:<30> log likelihood: -98244
    INFO:lda:<40> log likelihood: -97716
    INFO:lda:<50> log likelihood: -97499
    INFO:lda:<60> log likelihood: -97023
    INFO:lda:<70> log likelihood: -96947
    INFO:lda:<80> log likelihood: -96824
    INFO:lda:<90> log likelihood: -96794
    INFO:lda:<100> log likelihood: -96604
    INFO:lda:<110> log likelihood: -96537
    INFO:lda:<120> log likelihood: -96614
    INFO:lda:<130> log likelihood: -96392
    INFO:lda:<140> log likelihood: -96328
    INFO:lda:<150> log likelihood: -96437
    INFO:lda:<160> log likelihood: -96352
    INFO:lda:<170> log likelihood: -96205
    INFO:lda:<180> log likelihood: -96264
    INFO:lda:<190> log likelihood: -96213
    INFO:lda:<200> log likelihood: -96125
    INFO:lda:<210> log likelihood: -96110
    INFO:lda:<220> log likelihood: -96063
    INFO:lda:<230> log likelihood: -96064
    INFO:lda:<240> log likelihood: -95982
    INFO:lda:<250> log likelihood: -95948
    INFO:lda:<260> log likelihood: -96031
    INFO:lda:<270> log likelihood: -95931
    INFO:lda:<280> log likelihood: -96204
    INFO:lda:<290> log likelihood: -95973
    INFO:lda:<300> log likelihood: -95942
    INFO:lda:<310> log likelihood: -95976
    INFO:lda:<320> log likelihood: -95891
    INFO:lda:<330> log likelihood: -95890
    INFO:lda:<340> log likelihood: -95933
    INFO:lda:<350> log likelihood: -95955
    INFO:lda:<360> log likelihood: -95899
    INFO:lda:<370> log likelihood: -95929
    INFO:lda:<380> log likelihood: -95874
    INFO:lda:<390> log likelihood: -95917
    INFO:lda:<400> log likelihood: -95891
    INFO:lda:<410> log likelihood: -95902
    INFO:lda:<420> log likelihood: -95911
    INFO:lda:<430> log likelihood: -95939
    INFO:lda:<440> log likelihood: -95725
    INFO:lda:<450> log likelihood: -95829
    INFO:lda:<460> log likelihood: -95888
    INFO:lda:<470> log likelihood: -95874
    INFO:lda:<480> log likelihood: -95724
    INFO:lda:<490> log likelihood: -95825
    INFO:lda:<500> log likelihood: -95828
    INFO:lda:<510> log likelihood: -95997
    INFO:lda:<520> log likelihood: -95726
    INFO:lda:<530> log likelihood: -95765
    INFO:lda:<540> log likelihood: -95933
    INFO:lda:<550> log likelihood: -95802
    INFO:lda:<560> log likelihood: -95789
    INFO:lda:<570> log likelihood: -95771
    INFO:lda:<580> log likelihood: -95725
    INFO:lda:<590> log likelihood: -95823
    INFO:lda:<600> log likelihood: -95717
    INFO:lda:<610> log likelihood: -95785
    INFO:lda:<620> log likelihood: -95720
    INFO:lda:<630> log likelihood: -95669
    INFO:lda:<640> log likelihood: -95791
    INFO:lda:<650> log likelihood: -95703
    INFO:lda:<660> log likelihood: -95711
    INFO:lda:<670> log likelihood: -95744
    INFO:lda:<680> log likelihood: -95704
    INFO:lda:<690> log likelihood: -95745
    INFO:lda:<700> log likelihood: -95487
    INFO:lda:<710> log likelihood: -95621
    INFO:lda:<720> log likelihood: -95658
    INFO:lda:<730> log likelihood: -95654
    INFO:lda:<740> log likelihood: -95608
    INFO:lda:<750> log likelihood: -95532
    INFO:lda:<760> log likelihood: -95551
    INFO:lda:<770> log likelihood: -95476
    INFO:lda:<780> log likelihood: -95740
    INFO:lda:<790> log likelihood: -95609
    INFO:lda:<800> log likelihood: -95747
    INFO:lda:<810> log likelihood: -95653
    INFO:lda:<820> log likelihood: -95556
    INFO:lda:<830> log likelihood: -95549
    INFO:lda:<840> log likelihood: -95548
    INFO:lda:<850> log likelihood: -95573
    INFO:lda:<860> log likelihood: -95629
    INFO:lda:<870> log likelihood: -95546
    INFO:lda:<880> log likelihood: -95607
    INFO:lda:<890> log likelihood: -95565
    INFO:lda:<900> log likelihood: -95580
    INFO:lda:<910> log likelihood: -95646
    INFO:lda:<920> log likelihood: -95559
    INFO:lda:<930> log likelihood: -95721
    INFO:lda:<940> log likelihood: -95572
    INFO:lda:<950> log likelihood: -95585
    INFO:lda:<960> log likelihood: -95622
    INFO:lda:<970> log likelihood: -95584
    INFO:lda:<980> log likelihood: -95507
    INFO:lda:<990> log likelihood: -95639
    INFO:lda:<1000> log likelihood: -95640
    INFO:lda:<1010> log likelihood: -95530
    INFO:lda:<1020> log likelihood: -95475
    INFO:lda:<1030> log likelihood: -95548
    INFO:lda:<1040> log likelihood: -95582
    INFO:lda:<1050> log likelihood: -95599
    INFO:lda:<1060> log likelihood: -95636
    INFO:lda:<1070> log likelihood: -95608
    INFO:lda:<1080> log likelihood: -95446
    INFO:lda:<1090> log likelihood: -95560
    INFO:lda:<1100> log likelihood: -95567
    INFO:lda:<1110> log likelihood: -95523
    INFO:lda:<1120> log likelihood: -95522
    INFO:lda:<1130> log likelihood: -95568
    INFO:lda:<1140> log likelihood: -95500
    INFO:lda:<1150> log likelihood: -95467
    INFO:lda:<1160> log likelihood: -95545
    INFO:lda:<1170> log likelihood: -95549
    INFO:lda:<1180> log likelihood: -95623
    INFO:lda:<1190> log likelihood: -95611
    INFO:lda:<1200> log likelihood: -95652
    INFO:lda:<1210> log likelihood: -95549
    INFO:lda:<1220> log likelihood: -95694
    INFO:lda:<1230> log likelihood: -95534
    INFO:lda:<1240> log likelihood: -95486
    INFO:lda:<1250> log likelihood: -95569
    INFO:lda:<1260> log likelihood: -95556
    INFO:lda:<1270> log likelihood: -95457
    INFO:lda:<1280> log likelihood: -95458
    INFO:lda:<1290> log likelihood: -95435
    INFO:lda:<1300> log likelihood: -95512
    INFO:lda:<1310> log likelihood: -95516
    INFO:lda:<1320> log likelihood: -95429
    INFO:lda:<1330> log likelihood: -95567
    INFO:lda:<1340> log likelihood: -95511
    INFO:lda:<1350> log likelihood: -95602
    INFO:lda:<1360> log likelihood: -95408
    INFO:lda:<1370> log likelihood: -95465
    INFO:lda:<1380> log likelihood: -95600
    INFO:lda:<1390> log likelihood: -95592
    INFO:lda:<1400> log likelihood: -95553
    INFO:lda:<1410> log likelihood: -95588
    INFO:lda:<1420> log likelihood: -95556
    INFO:lda:<1430> log likelihood: -95604
    INFO:lda:<1440> log likelihood: -95543
    INFO:lda:<1450> log likelihood: -95588
    INFO:lda:<1460> log likelihood: -95573
    INFO:lda:<1470> log likelihood: -95579
    INFO:lda:<1480> log likelihood: -95461
    INFO:lda:<1490> log likelihood: -95620
    INFO:lda:<1499> log likelihood: -95451

    <lda.lda.LDA at 0x2aad38bcc90>

``` python
topic_word = model.topic_word_
n_top_words = 10
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))
```

    Topic 0: bane batman catwoman head tunnel hand pulls takes bat mercenary
    Topic 1: continuous ext blake bane gotham street dr pavel city truck
    Topic 2: wayne prisoner prison bane people dent cut gotham child prisoners
    Topic 3: blake gordon continuous ext selina gotham foley day cruiser pulls
    Topic 4: wayne alfred daggett fox miranda manor day bruce stock enterprises
