import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file1 = open(r"D:\mtech\2\ML\A3\dataset\data.txt", 'r')
data = []
x = []
y = []
input_x = []
output_y = []


def read_data():
    for i in file1.readlines():
        data.append(i.replace("\n", ""))
    
    for j in data:
        x.append(j.split(" ")[0])
        y.append(j.split(" ")[-1])
    for j in x:
        if j == 'x':
            continue
        input_x.append(float(j))
    for k in y:
        if k == 'y':
            continue
        output_y.append(float(k))

def plot_data(temp_flag):
    if temp_flag == 1:
        df = pd.DataFrame({'x_0':input_x[:15], 'y_o': output_y[:15]})
    elif temp_flag == 2:
        df = pd.DataFrame({'x_0':input_x[:20], 'y_o': output_y[:20]})
    elif temp_flag == 3:
        df = pd.DataFrame({'x_0':input_x[:80], 'y_o': output_y[:80]})
    elif temp_flag == 4:
        df = pd.DataFrame({'x_0':input_x[:100], 'y_o': output_y[:100]})
    plt.scatter(df['x_0'], df['y_o'])
    plt.show()
    return df

def hypothesis(X, theta):
    y1 = theta*X
    return np.sum(y1, axis=1)

def huber_loss(X, y, theta,reg=0):
    y1 = hypothesis(X, theta)
    error = np.absolute(y1-y)
    flag = error>delta
    total_cost = np.absolute(np.sum((~flag)*0.5*(error**2) - (flag*delta*error)-0.5*delta**2))
    l2_regularization_term = np.array([0.0])
    if reg:
        for i in range(0, len(theta)):
            l2_regularization_term[0] += np.absolute(theta[i])
        return total_cost+lam_bda*l2_regularization_term[0]
    return total_cost

def gradientDescent(X, y, theta, alpha, epoch, reg=0):
    J=[]
    k=0
    while k<epoch:
        y1 = hypothesis(X, theta)
        for c in range(0, len(X.columns)):
            if sum(abs(y1 - y)) <= delta:
                theta[c] = theta[c] - alpha*(sum((y1-y)* X.iloc[:, c])+lam_bda*l2_regularization_term[0])/m
            else:
                theta[c]= theta[c]- delta*alpha*(sum(X.iloc[:, c]*(y1 - y) / abs(y - y1))+lam_bda*l2_regularization_term)
        j = huber_loss(X, y, theta,reg)
        J.append(j)
        k += 1
    return J, theta   

def plot_results(itr_type, degree, data_points,J=0):
    y_hat = hypothesis(X_INPUT, theta)
    plt.figure() 
    plt.scatter(x=X_INPUT['x_0'],y= Y_OUTPUT)
    plt.scatter(x=X_INPUT['x_0'], y=y_hat)
    plt.savefig('results\huber_{}_degree_{}_points_{}.png'.format(itr_type,degree, data_points)) 
    plt.show()
    if J:
        plt.plot(J)
        plt.show()
        total = np.sum(J)
        J = J/total
        print(" Noise varience == ",np.var(J))
        print(J)
    
def test(x):
    df = plot_data(x)
    plot_results(5) 

print("started")
read_data()
df = plot_data(1)
X_INPUT = pd.DataFrame()
Y_OUTPUT = df['y_o']
X_INPUT['x_0'] = df['x_0']
X_INPUT['x_1'] = X_INPUT['x_0']**2
X_INPUT['x_2'] = X_INPUT['x_0']**3
X_INPUT['x_3'] = X_INPUT['x_0']**4
# X_INPUT['x_4'] = X_INPUT['x_0']**5
# X_INPUT['x_5'] = X_INPUT['x_0']**6
X_INPUT = pd.concat([pd.Series(1, index=X_INPUT.index, name='00'), X_INPUT], axis=1)


lam_bda = 0.05
delta = 20

m = len(X_INPUT)
l2_regularization_term = np.array([0.0])
noise = []

theta = np.array([0.0]*len(X_INPUT.columns))
print("training started")
J, theta = gradientDescent(X_INPUT, Y_OUTPUT, theta, 0.0001, 2000)
print("done")
plot_results("Train",4,20, J)

#Test with 5 data points
print("test started")
df = plot_data(2)
X_INPUT = pd.DataFrame()
Y_OUTPUT = df['y_o']
X_INPUT['x_0'] = df['x_0']
X_INPUT['x_1'] = X_INPUT['x_0']**2
X_INPUT['x_2'] = X_INPUT['x_0']**3
X_INPUT['x_3'] = X_INPUT['x_0']**4
# X_INPUT['x_4'] = X_INPUT['x_0']**5
# X_INPUT['x_5'] = X_INPUT['x_0']**6
X_INPUT = pd.concat([pd.Series(1, index=X_INPUT.index, name='00'), X_INPUT], axis=1)
plot_results("Test",4,20)
print("Test Done")

#Train with regularization
df = plot_data(1)
X_INPUT = pd.DataFrame()
Y_OUTPUT = df['y_o']
X_INPUT['x_0'] = df['x_0']
X_INPUT['x_1'] = X_INPUT['x_0']**2
X_INPUT['x_2'] = X_INPUT['x_0']**3
X_INPUT['x_3'] = X_INPUT['x_0']**4
# X_INPUT['x_4'] = X_INPUT['x_0']**5
# X_INPUT['x_5'] = X_INPUT['x_0']**6
X_INPUT = pd.concat([pd.Series(1, index=X_INPUT.index, name='00'), X_INPUT], axis=1)
theta = np.array([0.0]*len(X_INPUT.columns))
print("Training with L1 regularization")
J, theta = gradientDescent(X_INPUT, Y_OUTPUT, theta, 0.0001, 2000,1)
print("Training with L1 regularization is done")
plot_results("Reg_train",4,20, J)


#Test with 5 data points including regularization
print("test started 20 points with L1")
df = plot_data(2)
X_INPUT = pd.DataFrame()
Y_OUTPUT = df['y_o']
X_INPUT['x_0'] = df['x_0']
X_INPUT['x_1'] = X_INPUT['x_0']**2
X_INPUT['x_2'] = X_INPUT['x_0']**3
X_INPUT['x_3'] = X_INPUT['x_0']**4
# X_INPUT['x_4'] = X_INPUT['x_0']**5
# X_INPUT['x_5'] = X_INPUT['x_0']**6
X_INPUT = pd.concat([pd.Series(1, index=X_INPUT.index, name='00'), X_INPUT], axis=1)
plot_results("Test_reg",4,20)
print("Test Done")


############################################    20  data points completed #####################################################
################################################  100 Data Points Started  ####################################################


df = plot_data(3)
X_INPUT = pd.DataFrame()
Y_OUTPUT = df['y_o']

X_INPUT['x_0'] = df['x_0']
X_INPUT['x_1'] = X_INPUT['x_0']**2
X_INPUT['x_2'] = X_INPUT['x_0']**3
X_INPUT['x_3'] = X_INPUT['x_0']**4
X_INPUT['x_4'] = X_INPUT['x_0']**5
X_INPUT['x_5'] = X_INPUT['x_0']**6
# X_INPUT['x_6'] = X_INPUT['x_0']**7
# X_INPUT['x_7'] = X_INPUT['x_0']**8

X_INPUT = pd.concat([pd.Series(1, index=X_INPUT.index, name='00'), X_INPUT], axis=1)
print("Training for 100 points")
theta = np.array([0.0]*len(X_INPUT.columns))
J, theta = gradientDescent(X_INPUT, Y_OUTPUT, theta, 0.00001, 2000)
plot_results("Train_100",6,100, J)
print("done")

print("Testing for 100 points")
df = plot_data(4)
X_INPUT = pd.DataFrame()
Y_OUTPUT = df['y_o']

X_INPUT['x_0'] = df['x_0']
X_INPUT['x_1'] = X_INPUT['x_0']**2
X_INPUT['x_2'] = X_INPUT['x_0']**3
X_INPUT['x_3'] = X_INPUT['x_0']**4
X_INPUT['x_4'] = X_INPUT['x_0']**5
X_INPUT['x_5'] = X_INPUT['x_0']**6
# X_INPUT['x_6'] = X_INPUT['x_0']**7
# X_INPUT['x_7'] = X_INPUT['x_0']**8

X_INPUT = pd.concat([pd.Series(1, index=X_INPUT.index, name='00'), X_INPUT], axis=1)
plot_results("Test_100",6,100, J)
print("Test Done")

df = plot_data(3)
X_INPUT = pd.DataFrame()
Y_OUTPUT = df['y_o']

X_INPUT['x_0'] = df['x_0']
X_INPUT['x_1'] = X_INPUT['x_0']**2
X_INPUT['x_2'] = X_INPUT['x_0']**3
X_INPUT['x_3'] = X_INPUT['x_0']**4
X_INPUT['x_4'] = X_INPUT['x_0']**5
X_INPUT['x_5'] = X_INPUT['x_0']**6
# X_INPUT['x_6'] = X_INPUT['x_0']**7
# X_INPUT['x_7'] = X_INPUT['x_0']**8
print("L1 Reguralization Training for 100 points")
X_INPUT = pd.concat([pd.Series(1, index=X_INPUT.index, name='00'), X_INPUT], axis=1)
theta = np.array([0.0]*len(X_INPUT.columns))
J, theta = gradientDescent(X_INPUT, Y_OUTPUT, theta, 0.00001, 2000,1)
plot_results("Train_Reg",6,100, J)
print("done")

df = plot_data(4)
X_INPUT = pd.DataFrame()
Y_OUTPUT = df['y_o']
print("L1 Reguralization Testing for 100 points")
X_INPUT['x_0'] = df['x_0']
X_INPUT['x_1'] = X_INPUT['x_0']**2
X_INPUT['x_2'] = X_INPUT['x_0']**3
X_INPUT['x_3'] = X_INPUT['x_0']**4
X_INPUT['x_4'] = X_INPUT['x_0']**5
X_INPUT['x_5'] = X_INPUT['x_0']**6
# X_INPUT['x_6'] = X_INPUT['x_0']**7
# X_INPUT['x_7'] = X_INPUT['x_0']**8

X_INPUT = pd.concat([pd.Series(1, index=X_INPUT.index, name='00'), X_INPUT], axis=1)
plot_results("Test_Reg_100",6,100, J)

print("done")