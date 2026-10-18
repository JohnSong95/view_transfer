import numpy as np
import os
from tecplot import tecplot

m, n = 5, 4
XX = np.arange(m*n).reshape(m, n).astype(float)
YY = np.arange(m*n).reshape(m, n).astype(float)
Eta = np.random.rand(m, n)

save_name = r'C:\Users\A\Downloads\有效度-叶栅-主流湍流度5%\叶栅-主流湍流度5%\压力面无粗糙度\吹风比1.5\check_output.plt'
print('Saving to:', save_name)
tecplot(XX, YY, Eta, save_name)

print('File exists:', os.path.exists(save_name))
print('File size:', os.path.getsize(save_name))
with open(save_name, 'r') as f:
    lines = f.readlines()
print('First 5 lines:')
for l in lines[:5]:
    print(' ', l.rstrip())
print('Total lines:', len(lines))