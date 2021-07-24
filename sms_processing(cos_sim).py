from konlpy.tag import Okt
import numpy as np
from numpy.linalg import norm
from numpy import dot
import pandas as pd
import scipy.io
import re

#정규표현식 활용 
#링크 주소 형식 ex) www.naver.com
p = re.compile('[a-z].[a-z.].[a-z]')

okt = Okt()

def save_nouns(voicepishing): #링크가 존재하는지 확인, 단어별 분류
    url_chk = p.search(voicepishing[0]) 
    for i,s in enumerate(voicepishing):
        voicepishing[i] = ' '.join(okt.nouns(s))
    return url_chk
        

def make_features(voicepishing):
    features = []
    words = []
    # 보이스피싱 전체 문장에 대해 반복 알고리즘 -> 단어 단위 다시 저장
    for i in range(len(voicepishing)):
        # 위에서 나눈 명사들을 각각 공백기준으로 단어로 따로 나눈 중요한 단어 데이터들을 모아 단어리스트로 만들어줘
        word = voicepishing[i].split(" ")
        # 리스트를 자체를 더 큰 단어 리스트에 저장 (중첩리스트) -> [[],[],[]]
        words.append(word)
    
        # 보이스피싱 한문장에 대해서 단어개수만큼 반복 알고리즘 -> 필터링
        for i in range(len(word)):
            # 필터 알고리즘 - 중복없는 단어만 저장하도록
            if word[i] not in features:
                features.append(word[i])
    return features, words          
            
#특징 데이터에서 빈도수를 벡터화 (행렬로 )
def make_matrix(features, senctence, frequency_arr):
    frequency_list = []
    for feature in features:
        #해당 feat에 대한 freq초기화
        frequency = 0
        for word in senctence:
            if word == feature:
                frequency += 1
        # 한 feature가 끝났을 때 해당 feat에 대한 frequency저장
        frequency_list.append(frequency)
        
    frequency_arr.append(np.array(frequency_list))

def cos_sim(a,b):
    return dot(a,b) / (norm(a)* norm(b))


mat_file = scipy.io.loadmat('C:/Users/mejyo/Desktop/test2.mat')

s1 = "고객님 예약하신 물건 보냈드렸습니다. 클릭 후 조회부탁합니다. http://tinyurl.com/y9bjdgtl"

for t in range(len(mat_file['data'])):
    test = str(mat_file['data'][t][0][0])
    voicepishing = [s1, test]
    url_chk = save_nouns(voicepishing)
    features, words = make_features(voicepishing)
    frequency = []
    for i in range(len(voicepishing)):
        make_matrix(features,words[i], frequency)
    
    for i in range(len(voicepishing)):
        for j in range(i+1, len(voicepishing)):
            result = 0 if url_chk == None else 0.4
            value = cos_sim(frequency[i], frequency[j])
            result = result + value * 0.6
            if(result > 0.6):
                print(voicepishing[0])
                print(voicepishing[1])
                print("문장1 : " + s1 + "\n문장2 : " + test + "\n" + "사이의 유사도는 " + str(result) + "\n")                
                