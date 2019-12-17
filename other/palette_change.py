from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

#https://teratail.com/questions/187368?sort=1#1350
img = Image.open("1.png")
#インデックスカラーに変換
img = img.convert("P")
#data = list(img.getdata())
#print(data)
# カラーパレットにアクセスする。
palette = img.getpalette()
# リストの値は index=0 から順番に [R, G, B, R, G, B, ...]
palette = np.array(palette).reshape(-1, 3)
#print(palette)
# 入れ替え
#tmp = palette[2]
#palette[2] = palette[15]
#palette[15] = tmp
tmp = palette[1]#bin
palette[1] = palette[12]
palette[12] = [0,0,0]
tmp = palette[2]#can
palette[2] = palette[22]
palette[22] = [0,0,0]
tmp = palette[3]#plastic bottole
palette[3] = palette[24]
palette[24] = [0,0,0]
tmp = palette[4]#paper cup
palette[4] = palette[82]
palette[82] = [0,0,0]
print(palette)
"""
0[0,0,0]88[0,0,0]
1[102,0,0]11[51,0,0]
2[0,102,0]16[0,51,0]
3[102,102,0]17[51,51,0]18[102,51,0]23[51,102,0],60[102,102,51]
4[0,0,102]46[0,0,51]
"""
# パレット数の変更(★ここはつけない(元のプログラムにもこの処理はない))
#N_COLOR = len(palette)-251 #適当な数に減らす <--これを減らすと画像の色が失われるが、パレットの数は変わらない
#palette = palette[:N_COLOR]

# パレットの更新
palette = palette.reshape(-1).tolist()
img.putpalette(palette)
# 実データの更新
data = list(img.getdata())
print(data)
#11,16,17,18,23,46,47,52,53,60,88(12(1)),(22(2)),(24(3)),(82(4))
# この処理はやりたいことに従ってご自身で実装ください。
#for i,v in enumerate(data):
    #if v >= 5:
        #data[i] = 4 # とりあえず末尾の番号に振る？
#3項演算子+内含表記の合わせ技
#https://qiita.com/y__sama/items/a2c458de97c4aa5a98e7
update_data = [1 if i == 12 else 1 if i == 11 else 2 if i == 22 else 2 if i == 16 else 3 if i == 24 else 3 if i == 17 else 3 if i == 18 else 3 if i == 23 else 3 if i == 60 else 4 if i == 82 else 4 if i == 46 else 0 for i in data]
#update_data = [2 if i == 22 else i for i in data]

#print(update_data)
img.putdata(update_data)
img.save('chanege_palette_and_indexnumber.png')

#本当にputdataされているのか確認
palette = img.getpalette()
palette = np.array(palette).reshape(-1, 3)
#print(palette) paletteが更新されていることを確認済み
data = list(img.getdata())
#print(data)　#dataが更新されていることを確認済み
#print(img.mode) #P



