可以使用下面範例來tuning model  
https://github.com/google-research/bert/blob/master/predicting_movie_reviews_with_bert_on_tf_hub.ipynb  
imdb source data  
http://ai.stanford.edu/~amaas/data/sentiment/  
bert-tensorflow (1.0.1)

此範例有成功跑起來  
https://colab.research.google.com/github/google-research/bert/blob/master/predicting_movie_reviews_with_bert_on_tf_hub.ipynb#scrollTo=IhJSe0QHNG7U  

Errr, 字典新增會跳錯誤 , 猜測是 vocab_size問題
{
  "attention_probs_dropout_prob": 0.1, 
  "directionality": "bidi", 
  "hidden_act": "gelu", 
  "hidden_dropout_prob": 0.1, 
  "hidden_size": 768, 
  "initializer_range": 0.02, 
  "intermediate_size": 3072, 
  "max_position_embeddings": 512, 
  "num_attention_heads": 12, 
  "num_hidden_layers": 12, 
  "pooler_fc_size": 768, 
  "pooler_num_attention_heads": 12, 
  "pooler_num_fc_layers": 3, 
  "pooler_size_per_head": 128, 
  "pooler_type": "first_token_transform", 
  "type_vocab_size": 2, 
  "vocab_size": 119547
}  

方向可以重create model 裡面開始, 他有bertconig引述
bert.run_classifier.create_model()
bert_config = bert.modeling.BertConfig.from_json_file('D:\\multi_cased_L-12_H-768_A-12\\bert_config.json')


  
  
config = bert.modeling.BertConfig(vocab_size=32000, hidden_size=512,
    num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

  
載入tensorflow 模型方法
https://stackoverflow.com/questions/41265035/tensorflow-why-there-are-3-files-after-saving-the-model/41273348#41273348?newreg=59e468a7cac64e83a03ebffbe953fe85  
with tf.Session() as sess:  
    saver = tf.train.import_meta_graph('D:\\multi_cased_L-12_H-768_A-12\\bert_model.ckpt.meta')  
    saver.restore(sess, "D:\\multi_cased_L-12_H-768_A-12\\bert_model.ckpt")  
    tf.all_variables()  
    
    
  https://stackoverflow.com/questions/40601975/tensorflow-assign-requires-shapes-of-both-tensors-to-match-lhs-shape-20-rhs
