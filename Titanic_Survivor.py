import numpy as np
import pandas as pd
import math

df = pd.read_csv('Train.csv')
# df = pd.read_csv('trai.csv')
columnsToDrop = ['name','ticket','embarked','body','home.dest','cabin','boat']
# columnsToDrop = ['name','ticket','embarked','cabin','passengerId']
df = df.drop(columns=columnsToDrop)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])
df = df.fillna(df['age'].mean())
print(df.head())
x_train = df[['pclass','sex','age','sibsp','parch','fare','survived']]
# print(x_train.loc[1])

def entropy(col):
    counts = np.unique(col,return_counts=True)
    total = col.shape[0]
    ent = 0.0
    for ix in counts[1]:
        p = ix/total
        ent += (-1.0)*p*math.log2(p)
    return ent

def divideData(x_train,fkey,fval):
    x_right = pd.DataFrame([],columns=x_train.columns)
    x_left = pd.DataFrame([],columns=x_train.columns)

    for i in range(x_train.shape[0]):
        val = x_train[fkey].loc[i]
        temp = pd.DataFrame([x_train.loc[i].values],columns=x_train.columns)
        if(val>fval):
            x_right = pd.concat([x_right,temp],axis=0)
        else:
            x_left = pd.concat([x_left,temp],axis=0)
    return x_left,x_right

def info_gain(x_train,fkey,fval):
    
    x_left,x_right = divideData(x_train,fkey,fval)
    l = float(x_left.shape[0])/x_train.shape[0]
    r = float(x_right.shape[0])/x_train.shape[0]

    if(x_left.shape[0]==0 or x_right.shape[0]==0):
        return -1
    else:
        i_gain = entropy(x_train.survived) - (l*entropy(x_left.survived) + r*entropy(x_right.survived))
        return i_gain

class DecisionTree:
    def __init__(self,depth=0,max_depth=5):
        self.left = None
        self.right = None
        self.fkey = None
        self.fval = None
        self.max_depth = max_depth
        self.depth = depth
        self.target = None
    
    def train(self,x_train):
        features = ['pclass','sex','age','sibsp','parch','fare']
        gains = []
        for feature in features:
            gains.append(info_gain(x_train,feature,x_train[feature].mean()))
        self.fkey = features[np.argmax(gains)]
        self.fval = x_train[self.fkey].mean()
        print("The current feature is ",self.fkey)

        # Split data with fkey
        left_data,right_data = divideData(x_train,self.fkey,self.fval)
        left_data = left_data.reset_index(drop=True)
        right_data = right_data.reset_index(drop=True)

        if(left_data.shape[0]==0 or right_data.shape[0]==0):
            if x_train.survived.mean() >= 0.5:
                self.target = 1.0
            else:
                self.target = 0.0
            return

        if(self.depth >= self.max_depth):
            if x_train.survived.mean() >= 0.5:
                self.target = 1.0
            else:
                self.target = 0.0
            return   

        self.left = DecisionTree(depth=self.depth+1,max_depth=self.max_depth)
        self.left.train(left_data)
        self.right = DecisionTree(depth=self.depth+1,max_depth=self.max_depth)
        self.right.train(right_data)

        if (x_train.survived.mean()>0.5):
            self.target = 1.0
        else:
            self.target = 0.0
        return
        
    
    def predict(self,x_test):
        if(x_test[self.fkey]>self.fval):
            if(self.right is None):
                return self.target
            return self.right.predict(x_test)
        else:
            if(self.left is None):
                return self.target
            return self.left.predict(x_test)

dt = DecisionTree()
dt.train(x_train)

y_predicted = []
test_df = pd.read_csv("Test.csv")
test_df = test_df.drop(columns=columnsToDrop)
le = LabelEncoder()
test_df['sex'] = le.fit_transform(test_df['sex'])
test_df = test_df.fillna(test_df['age'].mean())
test_df = test_df[['pclass','sex','age','sibsp','parch','fare']]

for ix in range(test_df.shape[0]):
    y_predicted.append(dt.predict(test_df.loc[ix]))
answers = pd.DataFrame(np.array(y_predicted),columns=['survived'])
print(answers)
# Now testing data
