from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

data = pd.read_csv('apriori/transactions.csv')

data_grouped = data.groupby(['TransactionID', 'Item'])['Item'].count().unstack(fill_value=0)
data_binary = data_grouped.applymap(lambda x: 1 if x > 0 else 0)

frequent_itemsets = apriori(data_binary, min_support=0.1, use_colnames=True)

rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)

print(rules[['antecedents', 'consequents', 'confidence']])
