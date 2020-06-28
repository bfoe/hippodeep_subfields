# -*- mode: python -*-
a = Analysis(['model_apply_head_and_cortex.py'],
             excludes=['lib2to3', 'win32com', 'win32pdh','win32pipe','PIL'],
             hookspath=None,
             runtime_hooks=None)
             
#avoid warning
for d in a.datas:
    if '_C.cp37-win_amd64.pyd' in d[0]: 
        a.datas.remove(d)
        break

#manually include model /torchparams
a.datas += Tree('./torchparams', prefix='./torchparams') 

#manually include ANT's & templates
a.binaries += [('antsApplyTransforms.exe', 'antsApplyTransforms.exe', 'DATA')]
a.binaries += [('msvcp120.dll', 'msvcp120.dll', 'DATA')]
a.binaries += [('msvcr120.dll', 'msvcr120.dll', 'DATA')]
a.binaries += [('hippoboxL_128.nii.gz', 'hippoboxL_128.nii.gz', 'DATA')]
a.binaries += [('hippoboxR_128.nii.gz', 'hippoboxR_128.nii.gz', 'DATA')]

#remove unnecessary stuff        
a.datas = [x for x in a.datas if not ('tk8.6\msgs' in os.path.dirname(x[1]))]            
a.datas = [x for x in a.datas if not ('tk8.6\images' in os.path.dirname(x[1]))]            
a.datas = [x for x in a.datas if not ('tk8.6\demos' in os.path.dirname(x[1]))]            
a.datas = [x for x in a.datas if not ('tcl8.6\opt0.4' in os.path.dirname(x[1]))]            
a.datas = [x for x in a.datas if not ('tcl8.6\http1.0' in os.path.dirname(x[1]))]            
a.datas = [x for x in a.datas if not ('tcl8.6\encoding' in os.path.dirname(x[1]))]            
a.datas = [x for x in a.datas if not ('tcl8.6\msgs' in os.path.dirname(x[1]))]            
a.datas = [x for x in a.datas if not ('tcl8.6\tzdata' in os.path.dirname(x[1]))]

pyz = PYZ(a.pure)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          name='HippoDeep_subfields.exe',
          debug=False,
          strip=None,
          upx=False,
          console=True)         
        