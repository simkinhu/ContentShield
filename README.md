# ContentShield

基于Tensorflow的文本内容安全审核

使用GPT4 + claude3.5 sonnet一步一步引导写了使用Tensorflow来训练一个文本内容审核的模型代码；

模型训练需要大量的数据集去训练，在data下下载了部分数据,但是数据完全不够；

有数据集的小伙伴可以提供完善的数据集进行完整训练测速，目前功能感觉是正常的,目前的功能很简单；

都可以提PR进行更新优化，AI大模型遍地都是的今天需要一个可靠低成本的审核模型；



# 审核分类：
{
    "政治敏感": 0,
    "违禁违规": 1,
    "暴力极端": 2,
    "色情内容": 3,
    "侮辱性语言": 4,
    "恐怖内容": 5,
    "儿童不宜": 6,
    "欺诈行为": 7,
    "非法交易": 8,
    "网络暴力": 9,
    "自我伤害": 10,
    "仇恨歧视": 11,
    "不实信息": 12,
    "性骚扰": 13,
    "恶意推广": 14,
    "其它": 15
}

# 测试结果
以下是在消费级4070基于当前有的数据集进行了10论训练的结果

```这是一个多行代码示例
请输入要审核的文本（输入'quit'退出）: 陈水扁称做鬼也不放过检察官中新网2月20日电 据台湾TVBS报道，扁系“立委”高志鹏今天说，陈水扁已经两天没有进食，且最近会面时，陈水扁甚至说出“做鬼也不会放过检察官”。高志鹏还说，这几天会面时，陈水扁的心情确实是“十分悲愤”，因为想不到其它方法，只好绝食。不过陈水扁的律师在探视后接受采访时却说“不知道”，并指出这种举动没必要，因为全案已进入司法程序，应好好打官司。
1/1 [==============================] - 0s 15ms/step
分类结果: 其它
置信度: 0.74
--------------------------------------------------
请输入要审核的文本（输入'quit'退出）: 杨洁篪：中美元首将在伦敦金融峰会上会晤中新网2月21日电 (记者 周兆军)中国外交部长杨洁篪21日上午透露，国家主席胡锦涛将于四月初在伦敦举行的二十四国金融峰会上会见美国总统奥巴马。中美双方认为这次会见意义重大，将精心准备这次会见。杨洁篪21日上午与到访的美国国务卿希拉里·克林顿举行会谈。他是在会谈结束后举行的联合记者会上透露上述信息的。相关阅读：胡锦涛温家宝今日下午将分别会晤希拉里
1/1 [==============================] - 0s 15ms/step
分类结果: 政治敏感
置信度: 0.83
--------------------------------------------------
请输入要审核的文本（输入'quit'退出）: 胡锦涛会见希拉里 邀请奥巴马尽早访华新华网北京2月21日电 (记者钱彤)国家主席胡锦涛21日下午在人民大会堂会见了美国国务卿希拉里·克林顿。胡锦涛说，中美两国都是世界上有重要影响的国家。两国在事关世界和平与发展的重大问题上，具有广泛的共同利益，都肩负着重要责任。21世纪中美关系是世界上最重要的双边关系之一。在当前国际金融危机不断扩散和蔓延、各种全球性挑战日益突出的背景下，进一步深化和发展中美关系比以往任何时候都更为重要。中国政府始终从战略高度和长远角度看待中美关系，愿同美方一道，抓住机遇，共迎挑战，推动中美关系进一步向前发展。胡锦涛说，我热情邀请并欢迎奥巴马总统方便时尽早访华，期待着同他就双边关系和共同关心的问题深入交换意见，也期待着4月初在伦敦与他会晤。胡锦涛表示，中方愿与美方进一步加强在经贸、反恐、执法、科教、文卫、能源、环保等领域的交流与合作，以及在重大国际和地区问题上的磋商与协调，共同抵御国际金融危机冲击，有效应对气候变化等全球性挑战，推动两国关系健康顺利发展。希拉里·克林顿说，美中关系开启了积极合作的新时代，双方在众多领域和全球性问题上拥有广泛共同利益，美方愿进一步加强同中方在各领域合作。外交部部长杨洁篪等参加了会见。杨洁篪与希拉里举行会谈并共同会见中外记者戴秉国会见美国国务卿希拉里新华网北京2月21日电 (记者钱彤)国家主席胡锦涛21日下午在人民大会堂会见了美国国务卿希拉里·克林顿。胡锦涛说，中美两国都是世界上有重要影响的国家。两国在事关世界和平与发展的重大问题上，具有广泛的共同利益，都肩负着重要责任。21世纪中美关系是世界上最重要的双边关系之一。在当前国际金融危机不断扩散和蔓延、各种全球性挑战日益突出的背景下，进一步深化和发展中美关系比以往任何时候都更为重要。中国政府始终从战略高度和长远角度看待中美关系，愿同美方一道，抓住机遇，共迎挑战，推动中美关系进一步向前发展。胡锦涛说，我热情邀请并欢迎奥巴马总统方便时尽早访华，期待着同他就双边关系和共同关心的问题深入交换意见，也期待着4月初在伦敦与他会晤。胡锦涛表示，中方愿与美方进一步加强在经贸、反恐、执法、科教、文卫、能源、环保等领域的交流与合作，以及在重大国际和地区问题上的磋商与协调，共同抵御国际金融危机冲击，有效应对气候变化等全球性挑战，推动两国关系健康顺利发展。希拉里·克林顿说，美中关系开启了积极合作的新时代，双方在众多领域和全球性问题上拥有广泛共同利益，美方愿进一步加强同中方在各领域合作。外交部部长杨洁篪等参加了会见。杨洁篪与希拉里举行会谈并共同会见中外记者戴秉国会见美国国务卿希拉里
1/1 [==============================] - 0s 13ms/step
分类结果: 政治敏感
置信度: 0.76
--------------------------------------------------
请输入要审核的文本（输入'quit'退出）: 香港烟草税率大增五成每包香烟加价八元中广网香港2月26日消息（记者吴新伟）香港新公布的财政预算案给此间六十万烟民带来“坏消息”，该案建议实时大幅调高烟草税率百分之五十，每包香烟加价八元，烟草税收入可增至三十八亿元。大公网报道说，按此方案，香烟税款由现时每支约八角，增加至一元二角。烟民日后每买一包烟，即缴付了二十四元税，税款约占零售价超过六成。去年本港烟草税收入达三十亿元，若调高五成税率，烟草税收入可增至三十八亿元。政府强调，调高烟草税只是一项控烟措施，并非为增加收入。有研究显示，烟草税每增加一成，烟民的数目可减少百分之六点三，故加烟税是为烟民健康着想，间接可减少医疗开支，但目前难以评估成效，需视乎年轻烟民和烟草商的态度。政府增加烟草税以示控烟的决心，反吸烟团体一致表示支持。吸烟与健康委员会主席刘文文说，将烟草税大幅调高一半，有助减少烟民，尤其是年轻一辈，希望政府未来加强控烟的宣传，进一步减少烟民数目。港大公共卫生研究中心主管林大庆说，提高烟草税是一项迟来的措施，估计可减少二至三成烟民，但未来仍需加强控烟工作，如考虑调高定额罚款。不过，烟草业界大唱反调，认为加税对烟草消耗影响不大。烟草业联会行政秘书佐勒菲卡尔说，加税对总体市场不会有太大影响，因消费者可选择平价的走私烟，烟民不会因加税而减少。此外，有售卖香烟的报摊昨午起，实时调高零售价，所有牌子的香烟，每包一律加价八元，平均零售价加至逾三十元，最贵需要四十六元。但有便利店未及加烟价，市民涌至抢购一空。
1/1 [==============================] - 0s 12ms/step
分类结果: 其它
置信度: 0.80
--------------------------------------------------
请输入要审核的文本（输入'quit'退出）: 香港一所幼儿中心爆发流感13人受感染中新网3月3日电 据香港特区政府网站消息，香港卫生防护中心2日公布，石硖尾香港耀能协会石硖尾幼儿中心爆发流感，影响10名学童及3名职员，他们年龄介乎2-40岁。为预防疾病在校园内散播，卫生防护中心建议该幼儿中心3日起至9日停课一周，以便彻底消毒。患者2月25日开始出现发烧、咳嗽及喉咙痛等流感类病例感染的征状，12人曾求医，1名6岁男童入住联合医院，情况稳定。他的样本对甲型流感病毒呈阳性反应。卫护中心职员已视察幼儿中心，教导员工采取预防措施；并继续监察情况，以及提供健康教育。
1/1 [==============================] - 0s 11ms/step
分类结果: 其它
置信度: 0.75

