import urllib3
import shutil
http = urllib3.PoolManager()
url='http://ids2.copart.com/view/NAD110304/3df27398-5436-414c-be1f-b5284a752dcc.TIF'
path = '../loki.tif'
with http.request('GET', url, preload_content=False) as r, open(path, 'wb') as out_file:
    shutil.copyfileobj(r, out_file)
r.release_conn()