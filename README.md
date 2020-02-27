# Awesome Recommender Systems
A curated list of awesome resources about Recommender Systems.

声明：本列表包含部分网络上收集的推荐，稍后补上相应来源。

### 技术演进
- [推荐系统技术演进趋势：从召回到排序再到重排](https://zhuanlan.zhihu.com/p/100019681) - 2019
<br />

### 内容推荐
- [Bag of Tricks for Efficient Text Classification](https://www.semanticscholar.org/paper/Bag-of-Tricks-for-Efficient-Text-Classification-Joulin-Grave/023cc7f9f3544436553df9548a7d0575bb309c2e) - Facebook 开源的文本处理工具 fastText 背后原理。可以训练词嵌入向量，文本多分类，效率和线性模型一样，效果和深度学习一样，值得拥有。 - 2016

- [The Learning Behind Gmail Priority](https://www.semanticscholar.org/paper/The-Learning-Behind-Gmail-Priority-Aberdeen-Pacovsky/c32e8187d7a575432eee831294b5e2f67962d441) - 介绍了一种基于文本和行为给用户建模的思路，是信息流推荐的早期探索，Gmail 智能邮箱背后的原理。 - 2010

- [Recommender Systems Handbook(第三章，第九章)](https://book.douban.com/subject/3695850/) - 这本书收录了推荐系统很多经典论文，话题涵盖非常广，第三章专门讲内容推荐的基本原理，第九章是一个具体的基于内容推荐系统的案例。 - 2010

- [文本上的算法](https://github.com/yanxionglu/text_pdf?hmsr=toutiao.io&utm_medium=toutiao.io&utm_source=toutiao.io) - 介绍了文本挖掘中常用的算法，及基础概念。内容涉及概率论，信息论，文本分类，聚类，深度学习，推荐系统等。 - 2016

- [LDA 数学八卦](http://www.victoriawy.com/wp-content/uploads/2017/12/LDA%E6%95%B0%E5%AD%A6%E5%85%AB%E5%8D%A6.pdf) - 由浅入深地讲解 LDA 原理，对于实际 LDA 工具的使用有非常大的帮助。 - 2013
<br />

### 近邻推荐
- [Amazon.com recommendations: item-to-item collaborative filtering](https://www.semanticscholar.org/paper/Recommendations-Item-to-Item-Collaborative-Linden-Smith/4f3cc101a52c7601273d12769a7198f59519e8e6) - 介绍 Amazon 的推荐系统原理，主要是介绍 Item-Based 协同过滤算法。 - 2001

- [Slope One Predictors for Online Rating-Based Collaborative Filtering](https://arxiv.org/pdf/cs/0702144.pdf) - Slope One 算法。 - 2007

- [Item-Based Collaborative Filtering Recommendation Algorithms](http://files.grouplens.org/papers/www10_sarwar.pdf) - GroupLens 的研究团队对比了不同的 Item-to-Item 的推荐算法。 - 2001

- [Collaborative Recommendations Using Item-to-Item Similarity Mappings](https://patents.google.com/patent/US6266649B1/en) - 是的，Amazon 申请了 Item-Based 算法的专利，所以如果在美上市企业，小心用这个算法。 - 1998

- [Recommender Systems Handbook（第 4 章）](https://book.douban.com/subject/3695850/) - 第四章综述性地讲了近邻推荐，也就是基础协同过滤算法。 - 2010
<br />

### 矩阵分解
- [Matrix Factorization and Collaborative Filtering](http://101.96.10.63/acsweb.ucsd.edu/~dklim/mf_presentation.pdf) - 从 PCA 这种传统的数据降维方法讲起，综述了矩阵分解和协同过滤算法。矩阵分解也是一种降维方法。 - 2013

- [Factorization Meets the Neighborhood: a Multifaceted Collaborative Filtering Model](http://101.96.10.63/www.cs.rochester.edu/twiki/pub/Main/HarpSeminar/Factorization_Meets_the_Neighborhood-_a_Multifaceted_Collaborative_Filtering_Model.pdf) - 把矩阵分解和近邻模型融合在一起。 - 2008

- [BPR- Bayesian Personalized Ranking from Implicit Feedback](https://arxiv.org/pdf/1205.2618.pdf) - 更关注推荐结果的排序好坏，而不是评分预测精度，那么 BPR 模型可能是首选，本篇是出处。 - 2012

- [Collaborative Filtering for Implicit Feedback Datasets](http://yifanhu.net/PUB/cf.pdf) - 不同于通常矩阵分解处理的都是评分数据这样的显式反馈，本文介绍一种处理点击等隐式反馈数据的矩阵分解模型。 - 2008

- [Matrix Factorization Techniques For Recommender Systems](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf) - 本文是大神 Yehuda Koren 对矩阵分解在推荐系统中的应用做的一个普及性介绍，值得一读。 - 2009

- [The BellKor Solution to the Netflix Grand Prize](https://www.netflixprize.com/assets/GrandPrize2009_BPC_BellKor.pdf) - 也是一篇综述，或者说教程，针对 Netflix Prize 的。 - 2009
<br />

### 模型融合
- [Adaptive Bound Optimization for Online Convex Optimization](https://arxiv.org/pdf/1002.4908.pdf) - FTRL 是 CTR 预估常用的优化算法，本文介绍 FTRL 算法原理。 - 2010

- [在线最优化求解](https://github.com/wzhe06/Ad-papers/blob/master/Optimization%20Method/%E5%9C%A8%E7%BA%BF%E6%9C%80%E4%BC%98%E5%8C%96%E6%B1%82%E8%A7%A3(Online%20Optimization)-%E5%86%AF%E6%89%AC.pdf) - 是对 FTRL 的通俗版解说。 - 2014

- [Ad Click Prediction: a View from the Trenches](https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/41159.pdf) - FTRL 工程实现解读。 - 2013

- [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf) - 提出 FM 模型的论文，FM 用于 CTR 预估。 - 2010

- [Field-aware Factorization Machines for CTR Prediction](https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf) - FFM 模型，用于 CTR 预估。 - 2016

- [Practical Lessons from Predicting Clicks on Ads at Facebook](http://quinonero.net/Publications/predicting-clicks-facebook.pdf) - 提出了 LR + GBDT 的 CTR 预估模型。 - 2014

- [Wide & Deep Learning for Recommender Systems](https://arxiv.org/pdf/1606.07792.pdf) - 提出融合深度和宽度模型的Wide&Deep 模型，用于 CTR 预估。 - 2016
<br />

### Bandit 算法
- [Introduction to Bandits: Algorithms and Theory](https://sites.google.com/site/banditstutorial/) - 介绍 bandit 算法概念，理论和算法。分两部分分别对应小的选项候选集和大的选项候选集。 - 2011
<br />

### 深度学习
- [Deep Neural Networks for YouTube Recommendations](https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/45530.pdf) - 介绍 YouTube 视频推荐系统在深度神经网络上的尝试。能从中看到 wide&deep 模型的影子。 - 2016
<br />

### 其他实用算法
- [Detecting Near-Duplicates for Web Crawling](http://www2007.cpsc.ucalgary.ca/papers/paper215.pdf) - 在这篇论文中提出了 simhash 算法，用于大规模网页去重。 - 2007
<br />

### 常见架构
- [Activity Feeds Architecture](https://www.slideshare.net/danmckinley/etsy-activity-feeds-architecture) - 本文非常详细地介绍了社交动态信息流的架构设计细节。 - 2011
<br />

### 关键模块
- [Overlapping Experiment Infrastructure- More, Better, Faster Experimentation](https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/36500.pdf) - ABTest 实验平台的扛鼎之作，Google 出品，值得拥有。 - 2010
<br />

### 效果保证
- [Tutorial on Robustness of Recommender Systems](https://www.slideshare.net/neilhurley/tutorial-on-robustness-of-recommender-systems) - 本文非常详细讨论了对推荐系统的攻击和防护，并有实验模拟。 - 2011
<br />

### 冷启动
<br />

### 特征工程
<br />

### Embedding
- [EMBEDDING 在大厂推荐场景中的工程化实践](https://lumingdong.cn/engineering-practice-of-embedding-in-recommendation-scenario.html) - 2019
- [万物皆Embedding，从经典的word2vec到深度学习基本操作item2vec](https://zhuanlan.zhihu.com/p/53194407) - 2019
- [Embedding从入门到专家必读的十篇论文](https://zhuanlan.zhihu.com/p/58805184) - 2019
- [深度学习中不得不学的Graph Embedding方法](https://zhuanlan.zhihu.com/p/64200072) - 2019
- [从KDD 2018 Best Paper看Airbnb实时搜索排序中的Embedding技巧](https://zhuanlan.zhihu.com/p/55149901) - 2019
- [Airbnb如何解决Embedding的数据稀疏问题？](https://zhuanlan.zhihu.com/p/57313656) - 2019
<br />

### 多目标
<br />

### 强化学习
<br />

### 离线指标
<br />

### 线上评估
<br />

### Serving
<br />

## 大厂实战
## Netflix
- [Netflix大溃败：放弃算法崇拜，向好莱坞低头](https://mp.weixin.qq.com/s/1Jdb-8IdmnZmwofObhYanA) - 2018
- [你看到哪版电影海报，由算法决定：揭秘Netflix个性化推荐系统](https://mp.weixin.qq.com/s/lZ4FOOVIxsdKvfW45CYCnA) - 2017
- [Netflix与推荐系统](https://cloud.tencent.com/developer/article/1088952) - 2016

## Hulu
- [干货：从相关性到RNN，一家线上“租碟店”的视频推荐算法演进 | 公开课实录](https://mp.weixin.qq.com/s/KCEcgeiLfI5mKgFdo_Ri6Q) - 2018
- [公开课 | 看了10集《老友记》就被系统推荐了10季，Hulu如何用深度学习避免视频推荐的过拟合](https://mp.weixin.qq.com/s/4KbhzGPF9Jj6ylhKx86szw) - 2018

## YouTube
- [反思 Youtube 算法：个性化内容推荐能毁掉你的人格](https://36kr.com/p/5118920.html) - 2018
- [4篇YouTube推荐系统论文, 一起来看看别人家的孩子](https://medium.com/@yaoyaowd/4%E7%AF%87youtube%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E8%AE%BA%E6%96%87-%E4%B8%80%E8%B5%B7%E6%9D%A5%E7%9C%8B%E7%9C%8B%E5%88%AB%E4%BA%BA%E5%AE%B6%E7%9A%84%E5%AD%A9%E5%AD%90-b91279e03f83) - 2017
- [用DNN构建推荐系统-Deep Neural Networks for YouTube Recommendations论文精读](https://cloud.tencent.com/developer/article/1163931) - 2017
- [原来YouTube推荐系统的内幕是这样……](https://juejin.im/post/59a93438518825238b251cc2) - 2017
- [Youtube 短视频推荐系统变迁：从机器学习到深度学习](https://zhuanlan.zhihu.com/p/28244445) - 2017

## Google
- [谷歌推出TF-Ranking：用于排序算法的可扩展TensorFlow库](https://mp.weixin.qq.com/s/xg4yFIAqMe0bOEM8RRcWpQ) - 2018

## Microsoft
- [如何将知识图谱特征学习应用到推荐系统？](https://www.msra.cn/zh-cn/news/features/embedding-knowledge-graph-in-recommendation-system-ii) - 2018

## Amazon
- [重读经典 | 亚马逊“一键下单”的背后——个性化推荐系统的发展历程](https://cloud.tencent.com/developer/article/1109208) - 2018


## Kaggle
- [Kaggle大神带你上榜单Top2%：点击预测大赛纪实（上）](https://mp.weixin.qq.com/s/Xkt5c11wtxJgIvHC3p9wxA) - 2017
- [Kaggle大神带你上榜单Top2%：点击预测大赛纪实（下）](https://mp.weixin.qq.com/s/5TcykXfFw97mT_aR_ZDOkA) - 2017

## 美团
- [美团BERT的探索和实践](https://tech.meituan.com/2019/11/14/nlp-bert-practice.html) - 2019
- [独家揭秘！2.5亿用户的美团智能推荐平台是如何构建的？](https://mp.weixin.qq.com/s?__biz=MzU1NDA4NjU2MA==&mid=2247492434&idx=1&sn=99b360622511a185594e578182c3feab&scene=0#wechat_redirect) - 2018
- [深度学习在美团搜索广告排序的应用实践](https://tech.meituan.com/searchads_dnn.html) - 2018
- [美团“猜你喜欢”深度学习排序模型实践](https://tech.meituan.com/recommend_dnn.html) - 2018
- [深度学习在美团推荐平台排序中的运用](https://tech.meituan.com/dl.html) - 2017
- [旅游推荐系统的演进](https://tech.meituan.com/travel_recsys.html) - 2017
- [美团O2O排序解决方案——线上篇](https://tech.meituan.com/meituan_search_rank.html) - 2015
- [美团O2O排序解决方案——线下篇](https://tech.meituan.com/rerank_solution_offline.html) - 2015
- [美团推荐算法实践](https://tech.meituan.com/mt_recommend_practice.html) - 2015


## 今日头条
- [干货丨3分钟了解今日头条推荐算法原理（附视频+PPT）](https://mp.weixin.qq.com/s/_qtxPizf9LsXXYNYhhsvxA) - 2018
- [技术帖：解析今日头条公开的推荐算法](https://cloud.tencent.com/developer/article/1106562) - 2018
- [深度解密今日头条的个性化资讯推荐技术](https://cloud.tencent.com/developer/article/1042680) - 2017

## 抖音
- [抖音推荐系统冷启动](https://juejin.im/post/5b796b0b6fb9a019f47d0862) - 2018

## 微博
- [微博推荐系统架构揭秘：基于机器学习的个性化Push应用实践](https://cloud.tencent.com/developer/news/342140) - 2018
- [机器学习在热门微博推荐系统的应用](https://cloud.tencent.com/developer/article/1143169) - 2018
- [微博推荐算法如何设计](https://cloud.tencent.com/developer/article/1058317) - 2015

## 爱奇艺
- [爱奇艺个性化推荐排序实践](https://mp.weixin.qq.com/s?__biz=MzI0MjczMjM2NA==&mid=2247483872&idx=1&sn=db0fbb2bec0d4e68593f1b9bfc20a8b5) - 2017

## 搜狗
- [搜狗深度学习技术在广告推荐领域的应用](https://cloud.tencent.com/developer/article/1083548) - 2018

## 优酷
- [优酷视频基于用户兴趣个性化推荐的挑战和实践](https://www.infoq.cn/article/youku-recommendation-practice) - 2018
- [优酷视频精准推荐系统实践](https://myslide.cn/slides/248?vertical=1) - 2016

## 京东
- [京东个性化推荐系统实战（上）](https://cloud.tencent.com/developer/article/1080673) - 2018
- [京东个性化推荐系统实战（下）](https://cloud.tencent.com/developer/article/1080674) - 2018
- [深解京东个性化推荐系统演进史](http://www.woshipm.com/pd/882233.html) - 2017
- [京东推荐系统中的机器学习与大规模线上实验](https://cloud.tencent.com/developer/article/1083894) - 2017
- [京东推荐系统实践](https://myslide.cn/slides/9001?vertical=1) - 2015

## 阿里巴巴
- [阿里将 Transformer 用于淘宝电商推荐，效果优于 DIN 和谷歌 WDL](https://www.infoq.cn/article/OJvS7h8JXvD4XCW*qldw) - 2019
- [【阿里算法天才盖坤】解读阿里深度学习实践，CTR 预估、MLR 模型、兴趣分布网络等](https://mp.weixin.qq.com/s/9xw-FwWEK9VI2wNzc8EhPA) - 2018
- [阿里妈妈首次公开新一代智能广告检索模型，重新定义传统搜索框架](https://mp.weixin.qq.com/s/TrWwp-DBTrKqIT_Pfy_o5w) - 2018
- [论文Express | 淘宝广告是怎么优化的？阿里团队实时竞价系统策略](https://mp.weixin.qq.com/s/Tvz9s_O4RQcaxQ5u_c8brw) - 2018
- [个性化app推荐技术在手淘的全面应用](https://yq.aliyun.com/articles/603855?spm=a2c4e.11154873.tagmain.44.2f651257TuWQJG#) - 2018
- [深度丨110亿美金还不够，阿里使用这种AI手段创造更多广告收入（附PPT）丨CCF-GAIR 2017](https://www.leiphone.com/news/201707/t0AT4sIgyWS2QWVU.html) - 2017
- [千亿特征流式学习在大规模推荐排序场景的应用](https://yq.aliyun.com/articles/280235) - 2017

## 腾讯
- [深度技术解析，为什么说QQ音乐搜索体验做到了极致？](https://cloud.tencent.com/developer/article/1046806) - 2017
- [百亿级通用推荐系统实践](https://myslide.cn/slides/988?vertical=1) - 2016

## 美图
- [干货 | 美图个性化推荐的实践与探索](https://cloud.tencent.com/developer/article/1342091) - 2018

## 携程
- [干货 | 携程实时用户行为系统实践](https://cloud.tencent.com/developer/article/1063216) - 2017

## 饿了么
- [个性化推荐沙龙 | 饿了么推荐系统的从0到1（含视频）](https://cloud.tencent.com/developer/article/1063100) - 2017

## 58
- [58同城推荐系统架构设计与实现](https://cloud.tencent.com/developer/article/1048088) - 2016
- [从0开始做互联网推荐-以58转转为例](https://cloud.tencent.com/developer/article/1047908) - 2016

## 搜狐
- [搜狐视频个性化推荐架构设计和实践](https://myslide.cn/slides/4902?vertical=1) - 2016

## 百度
- [百度大规模推荐系统实践](https://myslide.cn/slides/3001?vertical=1) - 2016

## 知乎
- [进击的下一代推荐系统：多目标学习如何让知乎用户互动率提升 100%？](https://www.infoq.cn/article/hO8ZlIwDVsQmE1-V5oJ3) - 2019

## 参考来源：
- [推荐系统三十六式](https://time.geekbang.org/column/article/7204)
