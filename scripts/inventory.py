import json, pathlib, sys, platform, shutil, subprocess, hashlib
from importlib import metadata
root = pathlib.Path(r'C:\Users\Dala\Documents\New project\mmaction_rebuild')
envs=[]
for p in [pathlib.Path(r'D:\Anaconda'), *pathlib.Path(r'D:\Anaconda\envs').iterdir()]:
    site=p/'Lib/site-packages'
    if not site.is_dir(): continue
    packages=sorted([{'name':d.metadata['Name'],'version':d.version} for d in metadata.distributions(path=[str(site)])],key=lambda d:d['name'].lower())
    py=list((p/'conda-meta').glob('python-*.json'))
    envs.append({'path':str(p),'python_conda_record':[x.name for x in py],'packages':packages})
for e in envs:
    print(e['path'],e['python_conda_record'],[p for p in e['packages'] if p['name'].lower() in ['torch','torchvision','mmcv','mmcv-full','mmcv-lite','mmaction2','mmengine','numpy','opencv-python','decord']])
report={'host_python':sys.version,'platform':platform.platform(),'cpu':platform.processor(),'logical_cpus':__import__('os').cpu_count(),'disk':{x:dict(zip(['total','used','free'],shutil.disk_usage(x))) for x in ['C:\\','D:\\','E:\\']},'environments':envs,'gpu':subprocess.run(['nvidia-smi'],capture_output=True,text=True).stdout}
(root/'reports/host_inventory_before.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf8')
print('DISK',report['disk'])
