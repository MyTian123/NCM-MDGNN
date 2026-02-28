import numpy as np

a=np.load('XX.npy')
b=np.load('YY.npy')
mean=np.load('mean_list.npy')
std=np.load('std_list.npy')
mean_=[]
std_=[]

for i in range(a.shape[1]):
    mean_.append(np.mean(a[:,i]))
    std_.append(np.std(a[:,i])) 

np.save('mean_list_X.npy',np.array(mean_))
np.save('std_list_X.npy',np.array(std_))

for j in range(a.shape[1]):
    a[:,j] = (a[:,j]-mean_[j])/std_[j]

# for k in range(b.shape[0]):
#     print(b[k,:])
print(b[0,:])
print(b[1,:])

