from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import json
import tensorflow as tf
import util
#import pandas as pd
import numpy as np

'''
https://stackoverflow.com/questions/20290870/improving-the-extraction-of-human-names-with-nltk

import nltk
from nltk.tag.stanford import StanfordNERTagger
st = StanfordNERTagger('stanford-ner/english.all.3class.distsim.crf.ser.gz',
                       'stanford-ner/stanford-ner-2017-06-09.zip')
word= "The"
nltk.tokenize.word_tokenize(word)

tags = st.tag(word)
    for tag in tags:
        if tag[1] in ["PERSON", "LOCATION", "ORGANIZATION"]:
            print(tag) 
'''

if __name__ == "__main__":
  config = util.initialize_from_env()
  log_dir = config["log_dir"]

  # Input file in .jsonlines format.
  input_filename = sys.argv[2]
  #input_filename = "sample3.in.json"
  # Predictions will be written to this file in .jsonlines format.
  output_filename = sys.argv[3]
  #output_filename="output3.txt"
  model = util.get_model(config)
  saver = tf.train.Saver()

  with tf.Session() as session:
    model.restore(session)

    with open(output_filename, "w") as output_file:
      with open(input_filename) as input_file:
        for example_num, line in enumerate(input_file.readlines()):
          example = json.loads(line)
          tensorized_example = model.tensorize_example(example, is_training=False)
          feed_dict = {i:t for i,t in zip(model.input_tensors, tensorized_example)}
          _, _, _, top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores, emboutput = session.run(model.predictions, feed_dict=feed_dict)
          print(emboutput.shape)
          print(len(example['sentences'][0]))
          predicted_antecedents = model.get_predicted_antecedents(top_antecedents, top_antecedent_scores)
          example["predicted_clusters"], _ = model.get_predicted_clusters(top_span_starts, top_span_ends, predicted_antecedents)
          example["top_spans"] = list(zip((int(i) for i in top_span_starts), (int(i) for i in top_span_ends)))
          example['head_scores'] = []
          inputtext = example['sentences'][0].copy()
          for cluster in example["predicted_clusters"]:
            print(cluster)
            print(inputtext[cluster[-1][-1]])            
            reftext= inputtext[cluster[-2][0]:cluster[-2][-1]+1]
            if cluster[-2][0]==1:
              # 判断第一句话的第一个word是不是人名、地名、组织名等
              if inputtext[1]=="The":
                reftext[0]="the"    
            print(reftext)
            pop_l0=[]
            for i in range(len(reftext)):
              #把"##"分割的词拼接回去
              if "##" in reftext[i]:
                reftext[i]=''.join([reftext[i-1],reftext[i].replace("##",'')])
                pop_l0.append(reftext[i-1])
            #删除多余的词（前面已经增加拼接过的词，拼接前的词就删除掉）
            for poptext0 in pop_l0:
              reftext.remove(poptext0)
            pop_l1=[]
            for i in range(len(reftext)):
            #把"."切割的词拼回去
              if "."==reftext[i]:
                reftext[i]=''.join(reftext[(i-1):(i+2)])
                pop_l1.append(reftext[i-1])
                pop_l1.append(reftext[i+1])
            #删除多余的词（前面已经增加拼接过的词，拼接前的词就删除掉）
            for poptext1 in pop_l1:
              reftext.remove(poptext1)
            inputtext[cluster[-1][-1]] = ' '.join(reftext) #用预测的内容替换指示代词
          pop_l21=[]
          embedding_final_indice=[]
          for i in range(1,len(inputtext)-1):
            #把"##"分割的词拼接回去,举例来说，把"Di"和 "##m"拼成"Dim"
            if "##" in inputtext[i]:
              inputtext[i]=''.join([inputtext[i-1],inputtext[i].replace("##",'')])
              # 拼接回去后的词的embedding是原本词embedding的平均数
              emboutput[i-1]=(emboutput[i-1]+emboutput[i-1])/2
              pop_l21.append(inputtext[i-1])
            else:
              embedding_final_indice.append(i)
              
          #删除多余的词（前面已经增加拼接过的词，拼接前的词就删除掉）
          for poptext in pop_l21:
            print(poptext)
            if poptext in inputtext:
              inputtext.remove(poptext)
          pop_l22=[]
          for i in range(len(inputtext)):
          #把"."切割的词拼回去
            if "."==inputtext[i]:
              inputtext[i]=''.join(inputtext[(i-1):(i+2)])
              pop_l22.append(inputtext[i-1])
              pop_l22.append(inputtext[i+1])
              emboutput[i-1]=(emboutput[i-1]+emboutput[i]+emboutput[i+1])/3
              embedding_final_indice.pop(i)
              embedding_final_indice.pop(i+1)
          #删除多余的词（前面已经增加拼接过的词，拼接前的词就删除掉）
          for poptext1 in pop_l22:
            inputtext.remove(poptext1)
          # for j in range(2,len(inputtext)-1):
          #   inputtext[j]=inputtext[j].lower()
          example["output"]= ' '.join(inputtext[1:-1])
          print(example["output"])
          #example['inputtext_embedding']=emboutput
          #example['rawsentence_embedding']=emboutput[embedding_final_indice]
          #inputtext是加入mask"##"的words list
          #[["[CLS]", "I", "am", "hungry", ",", "please", "drive", "to", "the", "ne", "##ast", "Di", "##m", "Su", "##m", 
          #"restaurant", ".", "NO", "##UN", "and", "park", "next", "to", "it", "[SEP]"]], 
          np.savetxt("%s_inputtext_embedding.txt"%input_filename,emboutput)
          #pd.DataFrame(emboutput).to_csv("%s_inputtext_embedding.csv"%input_filename)
          # 删除mask以后，原本句子的word的embedding
          #["I am hungry, please drive to the neast Dim Sum restaurant.NOUN and park next to it"]
          #pd.DataFrame(emboutput[embedding_final_indice]).to_csv("%s_rawsentence_embedding.csv"%input_filename)
          np.savetxt("%s_rawsentence_embedding.txt"%input_filename,emboutput[embedding_final_indice])
          print(emboutput[embedding_final_indice].shape)
          output_file.write(json.dumps(example))
          output_file.write("\n")
          if example_num % 100 == 0:
            print("Decoded {} examples.".format(example_num + 1))
