# ASCM4ABSA
**ASCM4ABSA** - **A**spect-**s**pecific **C**ontext **M**odeling for **A**spect-**b**ased **S**entiment **A**nalysis
* Our code and proposed data (adversarial benchmark 'advABSA', which contains ARTS-OE-Lap and ARTS-OE-Res) for [NLPCC 2022](http://tcci.ccf.org.cn/conference/2022/index.php) paper titled "[Aspect-specific Context Modeling for Aspect-based Sentiment Analysis](https://arxiv.org/pdf/2207.08099.pdf)" 
* [Fang Ma](https://github.com/BD-MF), [Chen Zhang](https://genezc.github.io), [Bo Zhang](), and [Dawei Song](http://cs.bit.edu.cn/szdw/jsml/js/sdw/index.htm).

## AdvABSA

* Since there are only datasets for robustness tests in Aspect-based Sentiment Classification and is currently no dataset for robustness tests in Aspect-based Opinion Extraction, we propose an adversarial benchmark (advABSA) based on [xing2020tasty](https://aclanthology.org/2020.emnlp-main.292.pdf)'s datasets and methods. That is, the advABSA benchmark can be decomposed to two parts, where the first part is ARTS-SC for Aspect-based Sentiment Classification reused from [xing2020tasty](https://aclanthology.org/2020.emnlp-main.292.pdf) and the second part is ARTS-OE for Aspect-based Opinion Extraction crafted by us.


## Standard
* The standard folder is the standard datasets of SemEval 2014 for Aspect-based Sentiment Classification and Aspect-based Opinion Extraction, which contains data from laptop (Sem14-Lap-SC, Sem14-Lap-OE) and restaurant (Sem14-Rest-SC, Sem14-Rest-OE) domains.
