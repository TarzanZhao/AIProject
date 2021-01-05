import os
import shutil

oldpath = './data/stable_10_10_5'
newpath = './data/mix'

cnt = 0
for i in range(15):
    oldfile = os.path.join(oldpath, f'search-{i}')
    newfile = os.path.join(newpath, f'search-{cnt}')
    shutil.copy(oldfile, newfile)
    cnt += 1


