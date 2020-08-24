import json
import pandas as pd

def getFullText(jsonStruct):
    if 'extended_tweet' in jsonStruct:
        if 'full_text' in jsonStruct['extended_tweet']:
            return jsonStruct['extended_tweet']['full_text']
    elif 'text'in jsonStruct:
        return jsonStruct['text']
    else:
        return None

def processTweetArchive(archiveFile):
    with open(archiveFile, 'rt') as inF:
        allData = []
        for line in inF:
            fullTweetDict = json.loads(line)
            processedTweetDict = {}
            processedTweetDict['id'] = fullTweetDict['id']
            tweetText = ''            
            if 'retweeted_status' in fullTweetDict:
                tweetText += 'RT ' + getFullText(fullTweetDict['retweeted_status'])
            
            else:
                tweetText += getFullText(fullTweetDict)

            if 'quoted_status' in fullTweetDict:
                tweetText += ' [SEP] ' + getFullText(fullTweetDict['quoted_status'])
            processedTweetDict['text'] = tweetText
            
            processedTweetDict['user_screen_name'] = fullTweetDict['user']['screen_name']
            processedTweetDict['user_location'] = fullTweetDict['user']['location']
            if fullTweetDict['place'] is not None:
                processedTweetDict['country_code'] = fullTweetDict['place']['country_code']
            else:
                processedTweetDict['country_code'] = None

            
            allData.append(processedTweetDict)
        return allData


def loadCountryCodes():
    allCodes = []
    with open('resources/list_country_codes.txt') as inF:
        for line in inF:
            line = line.strip()
            if not line in ('BE', 'NL'):
                allCodes.append(line)
    return set(allCodes)

def loadGemeentenBE():
    allGemeenten = []
    with open('resources/lijst_gemeenten_be.csv') as inF:
        for line in inF:
            line = line.strip()
            gemeente, provincie = line.split('\t')
            allGemeenten.append(gemeente.lower())
    
    #both nl and be zwijndrecht
    #allGemeenten.remove('zwijndrecht')
    return set(allGemeenten)

def loadGemeentenNL():
    allGemeenten = []
    with open('resources/lijst_gemeenten_nl.csv') as inF:
        allLines = inF.readlines()
    #no header
    for line in allLines[2:]:
        line = line.strip()
        allData = line.split('\t')
        allGemeenten.append(allData[2].lower())

    #both nl and be zwijndrecht
    #allGemeenten.remove('zwijndrecht')
    return set(allGemeenten)
