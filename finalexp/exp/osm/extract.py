import subprocess, os
os.chdir('/root/roadplan/RoadPlan_remote')
data = subprocess.run(
    ['git', 'show', 'origin/main:数据/OSM走廊带障碍物/obstacles.npz'],
    capture_output=True).stdout
os.makedirs('/root/roadplan/finalexp/exp/osm', exist_ok=True)
with open('/root/roadplan/finalexp/exp/osm/obstacles.npz', 'wb') as f:
    f.write(data)
import numpy as np, collections
d = np.load('/root/roadplan/finalexp/exp/osm/obstacles.npz', allow_pickle=False)
print('bytes', len(data), '顶点', len(d['lines_lon']), '折线', len(d['kind']))
print('kind', collections.Counter(d['kind'].tolist()))
print('lon %.4f~%.4f lat %.4f~%.4f' % (
    d['lines_lon'].min(), d['lines_lon'].max(),
    d['lines_lat'].min(), d['lines_lat'].max()))
