import numpy as np
import pandas as pd

def save_large_array_to_excel(array, filename, chunk_size=10000):
    """
    将大型NumPy数组保存到Excel文件
    
    参数:
    array: NumPy数组，形状为(10016, 24)
    filename: 保存的文件名
    chunk_size: 分块处理的大小（避免内存问题）
    """
    # 将数组转换为DataFrame
    df = pd.DataFrame(array)
    
    # 添加列名（可选）
    df.columns = [f'Col_{i+1}' for i in range(array.shape[1])]
    
    # 保存到Excel
    df.to_excel(filename, index=False)
    
    print(f"数组已保存到 {filename}")
    print(f"数组形状: {array.shape}")

output_test=np.load('YY.npy').squeeze()

# mean_list_X=np.load('mean_list_X.npy').squeeze()
# std_list_X=np.load('std_list_X.npy').squeeze()
# output_test=np.load('output_test.npy').squeeze()
# output_test[:,0]=output_test[:,0]*std_list_X[0]+mean_list_X[0]
# output_test[:,1]=output_test[:,1]*std_list_X[1]+mean_list_X[1]
# for idx in range(output_test.shape[0]):
#     output_test[idx,0]=int(output_test[idx,0])
save_large_array_to_excel(output_test, 'resultsYY.xlsx')

print("数据已保存到 data.xlsx")