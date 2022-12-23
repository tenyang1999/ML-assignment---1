# ML-assignment---1
- 機器學習作業一：linear classifier 與 SVM實作
- 使用 crx & data進行預測
- 其中更使用sklearn的SVM進行比較
- 最終透過Accuracy去評分，比較模型效能

## 功能

- 能夠透過在clf資料夾中的各個分類器去執行分類預測。

## 安裝
- 本次執行上會需要用到的package

```python
pip install numpy
pip install pandas
pip install sklearn
```
將全部的model放置於資料夾clf中
- linearclassifier.py，為簡單線性分類器，於clf_acc.ipynb直接import即可套用
- linearclassifier_with_bias.py，為簡單線性分類器加入殘差項的版本，於clf_acc.ipynb直接import即可套用
- voted_perception.py，為投票型傳感器，於clf_acc.ipynb直接import即可套用
- SVM.py，為支持向量機，於clf_acc.ipynb直接import即可套用
- soft_margin_SVM.py，為支持向量機加入Slack variable，於clf_acc.ipynb直接import即可套用

## 使用方法
- tidy_crx.csv與tidy_data.csv為crx&data前處理後的檔案，詳細步驟寫於assignment.pdf中
- 執行在clf_acc.ipynb即可執行分類預測，將Q2-Q7中的所有分類器皆已導入。
