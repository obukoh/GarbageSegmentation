# GarbageSegmentation
The project that I classify the 4 kinds of garbages by u-net.

## 内容  
本研究では、unityでごみのデータセットを自作し、U-netを利用して、仮想空間におけるごみのセグメンテーションを行う。また、自作のデータセットにおいて、ごみ同士が重なり合っている画像を複数用意し、このような状況下でも、適切にごみのセグメンテーションができることを目標にする。

## データセット  
本研究では、unityで自作したデータセットを利用した。ごみはビン、缶、ペットボトル、紙ごみの4種類のごみをそれぞれ3種類ずつ用意した。ごみの分別保管の工程でラベルの色に依存せず、ごみの判別ができるよう物体の形のみを学習してセグメンテーションができる実験環境を用意した。このような条件下で、ビン、缶、ペットボトル、紙ごみが１つずつ写った画像を合計で16550枚作成した。そして、その中の85%をトレーニングデータ、残りの15%をテストデータとして、モデル評価を行った。
 
## 評価指標  
評価指標には、Mean Accuracyを用いた。Mean Accuracyとは、クラスごとの比率を正規化した指標である。本研究で自作したごみのデータセットは、実際の状況を想定して、ビン、缶、ペットボトル、紙ごみを均一の大きさに統一していない。この評価指標を用いた理由はそれぞれの物体が占める画素数が異なっていても、正しく物体の認識精度を評価するためである。
  
## 結果  
学習結果を図1に示す。左がテストデータの入力画像であり、右がテストデータの教師画像である。そして中央が結果の画像である。学習が進むごとに、適切にセグメンテーションできていることが確認できる。また、Epoch12のように隠れた物体も適切にセグメンテーションができることも確認できた。図2にAccuracyとLoss における学習過程のグラフを示す。テストデータに関して、Accuracyはおよそepoch 10で収束し始め、最終的に認識精度は、約98％となった。Lossに関しても最終的に、約0.01%で収束している。

#### 以下は、結果の画像である。  
![epoch_36](https://github.com/obukoh/GarbageSegmentation/blob/master/result/20200108_1441/image/test/epoch_36.png "epoch_36")  
![epoch_45](https://github.com/obukoh/GarbageSegmentation/blob/master/result/20200108_1441/image/test/epoch_45.png "epoch_45")  
![epoch_24](https://github.com/obukoh/GarbageSegmentation/blob/master/result/20200108_1441/image/test/epoch_24.png "epoch_24")  
左:入力画像　右:教師画像　中央:出力画像  

#### 以下は、AccuracyとLossのグラフである。  
![epoch_36](https://github.com/obukoh/GarbageSegmentation/blob/master/result/20200108_1441/learning/Accuracy.png "Accuracy")  
![epoch_36](https://github.com/obukoh/GarbageSegmentation/blob/master/result/20200108_1441/learning/Loss.png "Loss")
