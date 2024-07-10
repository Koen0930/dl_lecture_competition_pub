# DL基礎講座2024　最終課題「脳波分類」

## 概要
### 最終課題内容：3つのタスクから1つ選び，高い性能となるモデルを開発してください（コンペティション形式）
3つのタスクはそれぞれ以下の通りです．必ず**1つ**を選んで提出してください．
- 脳波分類（[`MEG-competition`](https://github.com/ailorg/dl_lecture_competition_pub/tree/MEG-competition)ブランチ）: 被験者が画像を見ているときの脳波から，その画像がどのクラスに属するかを分類する．
  - サンプル数: 訓練65,728サンプル，検証16,432サンプル，テスト16,432サンプル．
  - 入力: 脳波データ．
  - 出力: 1854クラスのラベル．
  - 評価指標: top-10 accuracy（モデルの予測確率トップ10に正解クラスが含まれているかどうか）．
- Visual Question Answering（VQA）（[`VQA-competition`](https://github.com/ailorg/dl_lecture_competition_pub/tree/VQA-competition)ブランチ）: 画像と質問から，回答を予測する．
  - サンプル数: 訓練19,873サンプル，テスト4,969サンプル．
  - 入力: 画像データ（RGB，サイズは画像によって異なる），質問文（サンプルごとに長さは異なる）．
  - 出力: 回答文（サンプルごとに長さは異なる）．
  - 評価指標: VQAでの評価指標（[こちら](https://visualqa.org/evaluation.html)を参照）を利用．
- Optical Flow Prediction from Event Camera (EventCamera)（[`event-camera-competition`](https://github.com/ailorg/dl_lecture_competition_pub/tree/event-camera-competition)ブランチ）: イベントカメラのデータから，Optical Flowを予測する．
  - サンプル数: 訓練7,800サンプル，テスト2,100サンプル．
  - 入力: イベントデータ（各時刻，どのピクセルで"log intensity"に変化があったかを記録）．
  - 出力: Optical Flow（連続フレーム間で，各ピクセルの動きをベクトルで表したもの）．
  - 評価指標: Average Endpoint Error（推定したOptical Flowと正解のOptical Flowのユークリッド距離）．

```bash
$ git clone git@github.com:[Github user name]/dl_lecture_competition_pub
```
3. `git checkout`を利用して自身が参加するコンペティションのブランチに切り替える．
- 以下のコマンドを利用して，参加したいコンペティションのブランチに切り替えてください．
- [Competition name]には`MEG-competition`（脳波分類タスク），`VQA-competition`（VQAタスク），`event-camera-competition`（EventCameraタスク）のいずれかが入ります．
```bash
$ cd dl_lecture_competition_pub
$ git checkout [Competition name]
```
4. README.mdの`環境構築`を参考に環境を作成します．
- README.mdにはconda，もしくはDockerを利用した環境構築の手順を記載しています．
5. README.mdの`ベースラインモデルを動かす`を参考に，ベースラインコードを実行すると，学習や予測が実行され，テストデータに対する予測である`submission.npy`が出力されます．

### 取り組み方
- ベースラインコードを書き換える形で，より性能の高いモデルの作成を目指してください．
  -  基本的には`main.py`などを書き換える形で実装してください．
  -  自分で1から実装しても構いませんが，ベースラインコードと同じ訓練データおよびテストデータを利用し，同じ形式で予測結果を出力してください．
- コンペティションでは，受講生の皆様に`main.py`の中身を書き換えて，より性能の高いモデルを作成していただき，予測結果(`submission.npy`)，工夫点レポート(`.pdf`)，実装したコードのリポジトリのURLを提出いただきます．
- 以下の条件を全て満たした場合に，最終課題提出と認めます．
   - 全ての提出物が提出されていること．
     - 注意：Omicampusで提出した結果が，レポートで書いた内容やコードと大きく異なる場合は，提出と認めない場合があります．
   - Omnicampusでの採点で各タスクのベースライン実装の性能を超えていること．
     - ベースライン性能は各タスクのブランチのREADMEを確認して下さい． 

### Githubへのpush方法
最終課題ではforkしたリポジトリをpublicに設定していただき，皆様のコードを評価に利用いたします．そのため，作成したコードをgithubへpushしていただく必要があります．

以下にgithubにその変更を反映するためのpushの方法を記載します．
1. `git add`
- 以下のように，`git add`に続けて変更を加えたファイル名を空白を入れて羅列します．
```bash
$ git add main.py hogehoge.txt hugahuga.txt
```
2. `git commit`
- `-m`オプションによりメモを残すことができます．その時の変更によって何をしたか，この後何をするのかなど記録しておくと便利です．
```bash
$ git commit -m "hogehoge"
``` 
3. `git push`
- branch nameには提出方法の手順3でcheckoutの後ろで指定したブランチ名を入力します．
```bash
$ git push origin [branch name]
```

## ベースラインモデルを動かす

### 訓練

```bash
python main.py

# オンラインで結果の可視化（wandbのアカウントが必要）
python main.py use_wandb=True
```

- `outputs/{実行日時}/`に重み`model_best.pt`と`model_last.pt`，テスト入力に対する予測`submission.npy`が保存されます．`submission.npy`をOmnicampusに提出することで，test top-10 accuracyが確認できます．

  - `model_best.pt`はvalidation top-10 accuracyで評価

- 訓練時に読み込む`config.yaml`ファイルは`train.py`，`run()`の`@hydra.main`デコレータで指定しています．新しいyamlファイルを作った際は書き換えてください．

- ベースラインは非常に単純な手法のため，改善の余地が多くあります（セクション「考えられる工夫の例」を参考）．そのため，**Omnicampusにおいてベースラインのtest accuracy=1.637%を超えた提出のみ，修了要件として認めることとします．**

### 評価のみ実行

- テストデータに対する評価のみあとで実行する場合．出力される`submission.npy`は訓練で最後に出力されるものと同じです．

```bash
python eval.py model_path={評価したい重みのパス}.pt
```

## データセット[[link](https://openneuro.org/datasets/ds004212/versions/2.0.0)]の詳細

- 1,854クラス，22,448枚の画像（1クラスあたり12枚程度）
  - クラスの例: airplane, aligator, apple, ...

- 各クラスについて，画像を約6:2:2の割合で訓練，検証，テストに分割

- 4人の被験者が存在し，どの被験者からのサンプルかは訓練に利用可能な情報として与えられる (`*_subject_idxs.pt`)．

### データセットのダウンロード

- [こちら](https://drive.google.com/drive/folders/1pgfVamCtmorUJTQejJpF8GhvwXa67rB9?usp=sharing)から`data.zip`をダウンロードし，`data/`ディレクトリに展開してください．

- 画像を事前学習などに用いる場合は，ドライブから`images.zip`をダウンロードし，任意のディレクトリで展開します．{train, val}_image_paths.txtのパスを使用し，自身でデータローダーなどを作成してください．

## タスクの詳細

- 本コンペでは，**被験者が画像を見ているときの脳波から，その画像がどのクラスに属するか**を分類します．

- 評価はtop-10 accuracyで行います．
  - モデルの予測確率トップ10に正解クラスが含まれているかどうか
  - つまりchance levelは10 / 1,854 ≒ 0.54%となります．

## 考えられる工夫の例

- 脳波の前処理
  - 配布したデータに対しては前処理が加えられていません．リサンプリングやフィルタリング，スケーリング，ベースライン補正など，波に対する基本的な前処理を試すことで性能の向上が見込まれます．
- 画像データを用いた事前学習
  - 本コンペのタスクは脳波のクラス分類ですが，配布してある画像データを脳波エンコーダの事前学習に用いることを許可します．
  - 例）CLIP [Radford+ 2021]
- 音声モデルの導入
  - 脳波と同じ波である音声を扱うアーキテクチャを用いることが有効であると知られています．
- 過学習を防ぐ正則化やドロップアウト
- 被験者情報の利用
  - 被験者ごとに脳波の特性が異なる可能性があるため，被験者情報を利用することで性能向上が見込まれます．
  - 例）Subject-specific layer [[Defossez+ 2022](https://arxiv.org/pdf/2208.12266)], domain adaptation
