#coding:utf-8
import pandas as pd
from sklearn import tree
import pydotplus
from graphviz import Digraph
from sklearn.externals.six import StringIO

"""
プロ野球データ(http://npb.jp/bis/players/)を対象に決定木を作成
Date:2016/6/23
Name:Daiki Kubo

"""

#文字列データをダミー変数化
def transDummy(idx, data, x):
	#Position
	dum_posi=pd.DataFrame(pd.get_dummies(data[idx]))
	# dum_posi.columns=['Infielder','Fielder','Pitcher','Catcher']
	# x=pd.concat([x,dum_posi],axis=1,join_axes=[x.index])
	x = pd.concat([x,dum_posi],axis=1)
	# print x
	return x

def genDTC(x, y, max_depth, random_state, criterion):
	clf = tree.DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=random_state)
	return clf.fit(x, y)

def graph(clf, categories, fname):
	dot_data = StringIO()
	tree.export_graphviz(clf, out_file=dot_data,feature_names=categories)
	graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
	graph.write_pdf(fname)


def genQustion(category):
	print '> 「' + category + '」に関連していますか? (はい / いいえ)'
	# print '> 「' + category + '」に関連していますか? (はい / いいえ / ひとつ戻る)'

def getResponse():
	return raw_input()

def printTargets(rest_targets, targets):
	print '----------------------'
	for idx in rest_targets:
		print targets[idx]
	print '----------------------'


def dialogue(clf, categories, targets, leafidx):
	position = [0]
	while position[-1] not in leafidx:
		genQustion(categories[clf.tree_.feature[position[-1]]])
		response = getResponse()
		if response == 'はい':
			position.append(clf.tree_.children_right[position[-1]])
		elif response == 'いいえ':
			position.append(clf.tree_.children_left[position[-1]])
		# elif response == '一つ戻る':
		# 	position.pop()
		# else:
		# 	print '「はい」もしくは「いいえ」でお願いします。。！	'

		# 結果の表示
		if clf.tree_.n_node_samples[position[-1]] < 10:
			rest_targets = [i for i, v in enumerate(clf.tree_.value[position[-1]][0]) if v == 1]
			printTargets(rest_targets, targets)


if __name__ == '__main__':
	labels = ['Number','Name','Position','Birthday','Height','Weight','PitchArm','SwingArm','League','Team']
	x = pd.DataFrame()
	data = pd.read_table('./training_data.tsv',names=labels)
	# data=transDummy(data)
	# for idx in ['Position', 'Team', 'PitchArm', 'SwingArm', 'League']:
	for idx in ['Position', 'Team', 'SwingArm', 'League']:
		x = transDummy(idx, data, x)

	y = pd.Series(data['Name'])

	# 決定木の分類器をCARTアルゴリズムで作成
	# clf = genDTC(x, y, max_depth=15, random_state=31, criterion='gini')
	clf = genDTC(x, y, max_depth=25, random_state=31, criterion='entropy')

	# # 決定木可視化
	graph(clf, x.columns, "akinator_graph.pdf")

	leafidx = [i for i, v in enumerate(clf.tree_.feature) if v == -2]	# リーフのインデックス取得
	# leafsamples = [clf.tree_.n_node_samples[idx] for idx in leafidx]

	# # インタラクション開始
	dialogue(clf, x.columns, y, leafidx)

	# print clf
	# print x
	# predicted = clf.predict(x)
	# print predicted

	# #モデルを用いて予測を実行する
	# print 'evaluate model ....'
	# #識別率を確認
	# result = sum(predicted == y) / len(y)
	# #result=1となった
	# print result

