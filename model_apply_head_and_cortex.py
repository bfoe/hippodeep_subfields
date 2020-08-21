import torch
import nibabel
import numpy as np
import os, sys, time
import scipy.ndimage
import torch.nn as nn
import torch.nn.functional as F
from numpy.linalg import inv
from numpy.linalg import det
if sys.platform=="win32": import psutil # this works for Windows
else: import resource # Unix specific package, does not exist for Windoze
from GetFilename import GetFilename
import warnings 
import subprocess

torch.set_num_threads(2)

OUTPUT_RES64 = False
OUTPUT_NATIVE = True
OUTPUT_DEBUG = False
saveprob = False 

code_labels = [ (0, 'unlabeled'), (1, 'parasubiculum'), (2, 'presubiculum'), (3, 'subiculum'),
                (4, 'CA1'), (5, 'CA3'), (6, 'CA4'), (7, 'GC-DG'), (8, 'HATA'), (9, 'fimbria'),
                (10, 'molecular_layer_HP'), (11, 'hippocampal_fissure'), (12, 'HP_tail')     ]

code_labels_L = [(c[0]+0,  "L_"+c[1]) for c in code_labels]
code_labels_R = [(c[0]+100,"R_"+c[1]) for c in code_labels]



if "-p" in sys.argv[1:]:
    args.remove("-p")
    saveprob = True
    
if "-d" in sys.argv[1:]:
    args.remove("-d")
    OUTPUT_DEBUG = True  

if "-h" in sys.argv[1:]:
    print ("Usage  :", sys.argv[0]," [ -d ] [ -p ] t1_mri_image")
    print ("Options:")
    print ("  -d   : debug mode")
    print ("  -p   : output the full probabilistic map for each label")
    print ("  -h   : this help")
    sys.exit(0)
      
if len(sys.argv[1:]) == 0:
  try: sys.argv.append(GetFilename())
  except:
    print("Need to pass one or more T1 image filename as argument")
    sys.exit(1)

class HeadModel(nn.Module):
    def __init__(self):
        super(HeadModel, self).__init__()
        self.conv0a = nn.Conv3d(1, 8, 3, padding=1)
        self.conv0b = nn.Conv3d(8, 8, 3, padding=1)
        self.bn0a = nn.BatchNorm3d(8)

        self.ma1 = nn.MaxPool3d(2)
        self.conv1a = nn.Conv3d(8, 16, 3, padding=1)
        self.conv1b = nn.Conv3d(16, 24, 3, padding=1)
        self.bn1a = nn.BatchNorm3d(24)

        self.ma2 = nn.MaxPool3d(2)
        self.conv2a = nn.Conv3d(24, 24, 3, padding=1)
        self.conv2b = nn.Conv3d(24, 32, 3, padding=1)
        self.bn2a = nn.BatchNorm3d(32)

        self.ma3 = nn.MaxPool3d(2)
        self.conv3a = nn.Conv3d(32, 48, 3, padding=1)
        self.conv3b = nn.Conv3d(48, 48, 3, padding=1)
        self.bn3a = nn.BatchNorm3d(48)


        self.conv2u = nn.Conv3d(48, 24, 3, padding=1)
        self.conv2v = nn.Conv3d(24+32, 24, 3, padding=1)
        self.bn2u = nn.BatchNorm3d(24)


        self.conv1u = nn.Conv3d(24, 24, 3, padding=1)
        self.conv1v = nn.Conv3d(24+24, 24, 3, padding=1)
        self.bn1u = nn.BatchNorm3d(24)


        self.conv0u = nn.Conv3d(24, 16, 3, padding=1)
        self.conv0v = nn.Conv3d(16+8, 8, 3, padding=1)
        self.bn0u = nn.BatchNorm3d(8)

        self.conv1x = nn.Conv3d(8, 4, 1, padding=0)

    def forward(self, x):
        x = F.elu(self.conv0a(x))
        self.li0 = x = F.elu(self.bn0a(self.conv0b(x)))

        x = self.ma1(x)
        x = F.elu(self.conv1a(x))
        self.li1 = x = F.elu(self.bn1a(self.conv1b(x)))

        x = self.ma2(x)
        x = F.elu(self.conv2a(x))
        self.li2 = x = F.elu(self.bn2a(self.conv2b(x)))

        x = self.ma3(x)
        x = F.elu(self.conv3a(x))
        self.li3 = x = F.elu(self.bn3a(self.conv3b(x)))

        x = F.interpolate(x, scale_factor=2, mode="nearest")

        x = F.elu(self.conv2u(x))
        x = torch.cat([x, self.li2], 1)
        x = F.elu(self.bn2u(self.conv2v(x)))

        self.lo1 = x
        x = F.interpolate(x, scale_factor=2, mode="nearest")

        x = F.elu(self.conv1u(x))
        x = torch.cat([x, self.li1], 1)
        x = F.elu(self.bn1u(self.conv1v(x)))

        x = F.interpolate(x, scale_factor=2, mode="nearest")
        self.la1 = x

        x = F.elu(self.conv0u(x))
        x = torch.cat([x, self.li0], 1)
        x = F.elu(self.bn0u(self.conv0v(x)))

        self.out = x = self.conv1x(x)
        x = torch.sigmoid(x)
        return x




class ModelAff(nn.Module):
    def __init__(self):
        super(ModelAff, self).__init__()
        self.convaff1 = nn.Conv3d(2, 16, 3, padding=1)
        self.maaff1 = nn.MaxPool3d(2)
        self.convaff2 = nn.Conv3d(16, 16, 3, padding=1)
        self.bnaff2 = nn.LayerNorm([32, 32, 32])

        self.maaff2 = nn.MaxPool3d(2)
        self.convaff3 = nn.Conv3d(16, 32, 3, padding=1)
        self.bnaff3 = nn.LayerNorm([16, 16, 16])

        self.maaff3 = nn.MaxPool3d(2)
        self.convaff4 = nn.Conv3d(32, 64, 3, padding=1)
        self.maaff4 = nn.MaxPool3d(2)
        self.bnaff4 = nn.LayerNorm([8, 8, 8])
        self.convaff5 = nn.Conv3d(64, 128, 1, padding=0)
        self.convaff6 = nn.Conv3d(128, 12, 4, padding=0)

        gsx, gsy, gsz = 64, 64, 64
        gx, gy, gz = np.linspace(-1, 1, gsx), np.linspace(-1, 1, gsy), np.linspace(-1,1, gsz)
        grid = np.meshgrid(gx, gy, gz) # Y, X, Z
        grid = np.stack([grid[2], grid[1], grid[0], np.ones_like(grid[0])], axis=3)
        netgrid = np.swapaxes(grid, 0, 1)[...,[2,1,0,3]]
        
        self.register_buffer('grid', torch.tensor(netgrid.astype("float32"), requires_grad = False))
        self.register_buffer('diagA', torch.eye(4, dtype=torch.float32))

    def forward(self, outc1):
        x = outc1
        x = F.relu(self.convaff1(x))
        x = self.maaff1(x)
        x = F.relu(self.bnaff2(self.convaff2(x)))
        x = self.maaff2(x)
        x = F.relu(self.bnaff3(self.convaff3(x)))
        x = self.maaff3(x)
        x = F.relu(self.bnaff4(self.convaff4(x)))
        x = self.maaff4(x)
        x = F.relu(self.convaff5(x))
        x = self.convaff6(x)

        x = x.view(-1, 3, 4)
        x = torch.cat([x, x[:,0:1] * 0], dim=1)
        self.tA = torch.transpose(x + self.diagA, 1, 2)

        wgrid = self.grid @ self.tA[:,None,None]
        gout = F.grid_sample(outc1, wgrid[...,[2,1,0]])
        return gout, self.tA

    def resample_other(self, other):
        with torch.no_grad():
            wgrid = self.grid @ self.tA[:,None,None]
            gout = F.grid_sample(other, wgrid[...,[2,1,0]])
            return gout

class HippoModel2(nn.Module):
  def __init__(self):
    super(HippoModel2, self).__init__()
    self.conv0a = nn.Conv3d(1, 12, 3, padding=1)
    self.conv0ap = nn.Conv3d(12, 12, 3, padding=1)
    self.conv0b = nn.Conv3d(12, 12, 3, padding=1)
    self.bn0a = nn.BatchNorm3d(12)

    self.ma1 = nn.MaxPool3d(2)
    self.conv1a = nn.Conv3d(12, 12, 3, padding=1)
    self.conv1ap = nn.Conv3d(12, 12, 3, padding=1)
    self.conv1b = nn.Conv3d(12, 12, 3, padding=1)
    self.bn1a = nn.BatchNorm3d(12)

    self.ma2 = nn.MaxPool3d(2)
    self.conv2a = nn.Conv3d(12, 16, 3, padding=1)
    self.conv2ap = nn.Conv3d(16, 16, 3, padding=1)
    self.conv2b = nn.Conv3d(16, 16, 3, padding=1)
    self.bn2a = nn.BatchNorm3d(16)

    self.ma3 = nn.MaxPool3d(2)
    self.conv3a = nn.Conv3d(16, 32, 3, padding=1)
    self.conv3ap = nn.Conv3d(32, 32, 3, padding=1)
    self.conv3b = nn.Conv3d(32, 24, 3, padding=1)
    self.bn3a = nn.BatchNorm3d(24)

    # up

    self.conv2u = nn.Conv3d(24, 16, 3, padding=1)
    self.conv2up = nn.Conv3d(16, 16, 3, padding=1)
    self.bn2u = nn.BatchNorm3d(16)
    self.conv2v = nn.Conv3d(16+16, 16, 3, padding=1)

    # up

    self.conv1u = nn.Conv3d(16, 12, 3, padding=1)
    self.conv1up = nn.Conv3d(12, 12, 3, padding=1)
    self.bn1u = nn.BatchNorm3d(12)
    self.conv1v = nn.Conv3d(12+12, 12, 3, padding=1)

    # up

    self.conv0u = nn.Conv3d(12, 12, 3, padding=1)
    self.conv0up = nn.Conv3d(12, 12, 3, padding=1)
    self.bn0u = nn.BatchNorm3d(12)
    self.conv0v = nn.Conv3d(12+12, 12, 3, padding=1)

    self.conv1x = nn.Conv3d(12, 13, 1, padding=0)

  def forward(self, x):
    x = F.elu(self.conv0a(x))
    self.li0 = x = F.elu(self.conv0ap (F.elu(self.bn0a(self.conv0b(x))) ))

    x = self.ma1(x)
    x = F.elu(self.conv1ap( F.elu(self.conv1a(x)) ))
    self.li1 = x = F.elu(self.bn1a(self.conv1b(x)))

    x = self.ma2(x)
    x = F.elu(self.conv2ap( F.elu(self.conv2a(x)) ))
    self.li2 = x = F.elu(self.bn2a(self.conv2b(x)))

    x = self.ma3(x)
    x = F.elu(self.conv3ap( F.elu(self.conv3a(x)) ))
    self.li3 = x = F.elu(self.bn3a(self.conv3b(x)))

    x = F.interpolate(x, scale_factor=2, mode="nearest")

    x = F.elu(self.conv2up( F.elu(self.bn2u(self.conv2u(x))) ))
    x = torch.cat([x, self.li2], 1)
    x = F.elu(self.conv2v(x))

    self.lo1 = x
    x = F.interpolate(x, scale_factor=2, mode="nearest")

    x = F.elu(self.conv1up( F.elu(self.bn1u(self.conv1u(x))) ))
    x = torch.cat([x, self.li1], 1)
    x = F.elu(self.conv1v(x))

    x = F.interpolate(x, scale_factor=2, mode="nearest")
    self.la1 = x

    x = F.elu(self.conv0up( F.elu(self.bn0u(self.conv0u(x))) ))
    x = torch.cat([x, self.li0], 1)
    x = F.elu(self.conv0v(x))

    self.out = x = self.conv1x(x)
    #x = torch.sigmoid(x)
    return x            
            

def bbox_world(affine, shape):
    s = shape[0]-1, shape[1]-1, shape[2]-1
    bbox = [[0,0,0], [s[0],0,0], [0,s[1],0], [0,0,s[2]], [s[0],s[1],0], [s[0],0,s[2]], [0,s[1],s[2]], [s[0],s[1],s[2]]]
    w = affine @ np.column_stack([bbox, [1]*8]).T
    return w.T

bbox_one = np.array([[-1,-1,-1,1], [1, -1, -1, 1], [-1, 1, -1, 1], [-1, -1, 1, 1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1], [1,1,1,1]])

affine64_mni = \
np.array([[  -2.85714293,   -0.        ,    0.        ,   90.        ],
          [  -0.        ,    3.42857146,   -0.        , -126.        ],
          [   0.        ,    0.        ,    2.85714293,  -72.        ],
          [   0.        ,    0.        ,    0.        ,    1.        ]])


try: scriptpath = sys._MEIPASS # when running frozen with pyInstaller 
except: scriptpath = os.path.dirname(os.path.realpath(__file__))

warnings.filterwarnings("ignore") # disable the following torch warning
#UserWarning: Default grid_sample and affine_grid behavior will be changed to 
#align_corners=False from 1.4.0. See the documentation of grid_sample for details.
#"Default grid_sample and affine_grid behavior will be changed "

device = torch.device("cpu")
net = HeadModel()
net.to(device)
net.load_state_dict(torch.load(os.path.normpath(scriptpath + "/torchparams/params_head_00075_00000.pt"), map_location=device))
net.eval()

netAff = ModelAff()
netAff.load_state_dict(torch.load(os.path.normpath(scriptpath + "/torchparams/paramsaffineta_00079_00000.pt"), map_location=device), strict=False)
netAff.to(device)
netAff.eval()

nibabel.openers.Opener.default_compresslevel = 9


for fname in sys.argv[1:]:
    Ti = time.time()
    try:
        print("Loading image " + fname)
        outfilename = fname
        for suffix in ".mnc .gz .nii .img .hdr .mgz .mgh".split():
            outfilename = outfilename.replace(suffix, "")
        outfilename = outfilename + "_tiv.nii.gz"
        img = nibabel.load(fname)
        if type(img) is nibabel.nifti1.Nifti1Image:
            img._affine = img.get_qform() # for ANTs compatibility        
    except:
        open(fname + ".warning.txt", "a").write("can't open the file\n")
        print("Warning: can't open file. Skip")
        continue

    d = img.get_data(caching="unchanged").astype(np.float32)
    while len(d.shape) > 3:
        print("Warning: this looks like a timeserie. Averaging it")
        open(fname + ".warning.txt", "a").write("dim not 3. Averaging last dimension\n")
        d = d.mean(-1)

    d = (d - d.mean()) / d.std()

    o1 = nibabel.orientations.io_orientation(img.affine)
    o2 = np.array([[ 0., -1.], [ 1.,  1.], [ 2.,  1.]]) # We work in LAS space (same as the mni_icbm152 template)
    trn = nibabel.orientations.ornt_transform(o1, o2) # o1 to o2 (apply to o2 to obtain o1)
    trn_back = nibabel.orientations.ornt_transform(o2, o1)    

    revaff1 = nibabel.orientations.inv_ornt_aff(trn, (1,1,1)) # mult on o1 to obtain o2
    revaff1i = nibabel.orientations.inv_ornt_aff(trn_back, (1,1,1)) # mult on o2 to obtain o1

    aff_orig64 = np.linalg.lstsq(bbox_world(np.identity(4), (64,64,64)), bbox_world(img.affine, img.shape[:3]), rcond=-1)[0].T
    voxscale_native64 = np.abs(np.linalg.det(aff_orig64))
    revaff64i = nibabel.orientations.inv_ornt_aff(trn_back, (64,64,64))
    aff_reor64 = np.linalg.lstsq(bbox_world(revaff64i, (64,64,64)), bbox_world(img.affine, img.shape[:3]), rcond=-1)[0].T
    
    wgridt = (netAff.grid @ torch.tensor(revaff1i, device=device, dtype=torch.float32))[None,...,[2,1,0]]
    d_orr = F.grid_sample(torch.as_tensor(d, dtype=torch.float32, device=device)[None,None], wgridt)

    if OUTPUT_DEBUG:
        nibabel.Nifti1Image(np.asarray(d_orr[0,0].cpu()), aff_reor64).to_filename(outfilename.replace("_tiv", "_orig_b64"))

## Head priors
    T = time.time()
    with torch.no_grad():
        out1t = net(d_orr)
    out1 = np.asarray(out1t.cpu())
    #print("Head Inference in ", time.time() - T)

    ## Output head priors
    scalar_output = []
    scalar_output_report = []

    if OUTPUT_NATIVE:
        # wgridt for native space
        gsx, gsy, gsz = img.shape[:3]
        # this is a big array, so use float16
        gx, gy, gz = np.linspace(-1, 1, gsx, dtype="f2"), np.linspace(-1, 1, gsy, dtype="f2"), np.linspace(-1,1, gsz, dtype="f2")
        grid = np.meshgrid(gx, gy, gz) # Y, X, Z
        grid = np.stack([grid[2], grid[1], grid[0], np.ones_like(grid[0])], axis=3)
        nativegrid1 = np.swapaxes(grid, 0, 1)[...,[2,1,0,3]]
        del grid
        wgridt = torch.as_tensor((nativegrid1 @ inv(revaff1i))[None,...,[2,1,0]], device=device, dtype=torch.float32)
        del nativegrid1

    # brain mask
    output = out1[0,0].astype("float32")

    out_cc, lab = scipy.ndimage.label(output > .01)
    #output *= (out_cc == np.bincount(out_cc.flat)[1:].argmax()+1)
    brainmask_cc = torch.tensor(output, device=device)

    vol = (output[output > .5]).sum() * voxscale_native64
    if OUTPUT_DEBUG:
        print(" Estimated intra-cranial volume (mm^3): %d" % vol)
    scalar_output.append(vol)
    scalar_output_report.append(vol)
     
    if OUTPUT_RES64:
        out = (output.clip(0, 1) * 255).astype("uint8")
        nibabel.Nifti1Image(out, aff_reor64, img.header).to_filename(outfilename.replace("_tiv", "_tissues%d_b64" % 0))
    if OUTPUT_NATIVE:
        dnat = np.asarray(F.grid_sample(torch.as_tensor(output, dtype=torch.float32, device=device)[None,None], wgridt).cpu())[0,0]
        #nibabel.Nifti1Image(dnat, img.affine).to_filename(outfilename.replace("_tiv", "_tissues%d" % 0))
        nibabel.Nifti1Image((dnat > .5).astype("uint8"), img.affine).to_filename(outfilename.replace("_tiv", "_brain_mask"))
        vol = (dnat > .5).sum() * np.abs(np.linalg.det(img.affine))
        print(" Estimated intra-cranial volume (mm^3) (native space): %d" % vol)
        scalar_output.append(vol)
        del dnat

    # cerebrum mask
    output = out1[0,2].astype("float32")

    out_cc, lab = scipy.ndimage.label(output > .01)
    output *= (out_cc == np.bincount(out_cc.flat)[1:].argmax()+1)

    vol = (output[output > .5]).sum() * voxscale_native64
    if OUTPUT_DEBUG:
        print(" Estimated cerebrum volume (mm^3): %d" % vol)
    scalar_output.append(vol)

    if OUTPUT_RES64:
        out = (output.clip(0, 1) * 255).astype("uint8")
        nibabel.Nifti1Image(out, aff_reor64, img.header).to_filename(outfilename.replace("_tiv", "_tissues%d_b64" % 2))
    if OUTPUT_NATIVE:
        dnat = np.asarray(F.grid_sample(torch.as_tensor(output, dtype=torch.float32, device=device)[None,None], wgridt).cpu()[0,0])
        #nibabel.Nifti1Image(dnat, img.affine).to_filename(outfilename.replace("_tiv", "_tissues%d" % 2))
        nibabel.Nifti1Image((dnat > .5).astype("uint8"), img.affine).to_filename(outfilename.replace("_tiv", "_cerebrum_mask"))
        vol = (dnat > .5).sum() * np.abs(np.linalg.det(img.affine))
        print(" Estimated cerebrum volume (mm^3) (native space): %d" % vol)
        scalar_output.append(vol)
        del dnat

    # cortex
    output = out1[0,1].astype("float32")
    output[output < .01] = 0
    if OUTPUT_RES64:
        out = (output.clip(0, 1) * 255).astype("uint8")
        nibabel.Nifti1Image(out, aff_reor64, img.header).to_filename(outfilename.replace("_tiv", "_tissues%d_b64" % 1))
    if (OUTPUT_NATIVE and OUTPUT_DEBUG):
        dnat = np.asarray(F.grid_sample(torch.as_tensor(output, dtype=torch.float32, device=device)[None,None], wgridt).cpu()[0,0])
        nibabel.Nifti1Image(dnat, img.affine).to_filename(outfilename.replace("_tiv", "_tissues%d" % 1))
        del dnat


## MNI affine
    T = time.time()
    with torch.no_grad():
        wc1, tA = netAff(out1t[:,[1,3]] * brainmask_cc)

    wnat = np.linalg.lstsq(bbox_world(img.affine, img.shape[:3]), bbox_one @ revaff1, rcond=-1)[0]
    wmni = np.linalg.lstsq(bbox_world(affine64_mni, (64,64,64)), bbox_one, rcond=-1)[0]
    M = (wnat @ inv(np.asarray(tA[0].cpu())) @ inv(wmni)).T
    # [native world coord] @ M.T -> [mni world coord] , in LAS space

    if OUTPUT_DEBUG:
        # Output MNI, mostly for debug, save in box64, uint8
        out2 = np.asarray(wc1.to("cpu"))
        out2 = np.clip((out2 * 255), 0, 255).astype("uint8")
        nibabel.Nifti1Image(out2[0,0], affine64_mni).to_filename(outfilename.replace("_tiv", "_mniwrapc1"))
        del out2
    if 0:
        out2r = np.asarray(netAff.resample_other(d_orr).cpu())
        out2r = (out2r - out2r.min()) * 255 / out2r.ptp()
        nibabel.Nifti1Image(out2r[0,0].astype("uint8"), affine64_mni).to_filename(outfilename.replace("_tiv", "_mniwrap"))
        del out2r


    # output an ANTs-compatible matrix (AntsApplyTransforms -t)
    f3 = np.array([[1, 1, -1, -1],[1, 1, -1, -1], [-1, -1, 1, 1], [1, 1, 1, 1]]) # ANTs LPS
    MI = inv(M) * f3
    txt = """#Insight Transform File V1.0\nTransform: AffineTransform_float_3_3\nFixedParameters: 0 0 0\nParameters: """
    txt += " ".join(["%4.6f %4.6f %4.6f" % tuple(x) for x in MI[:3,:3].tolist()]) + " %4.6f %4.6f %4.6f\n" % (MI[0,3], MI[1,3], MI[2,3])
    open(outfilename.replace("_tiv.nii.gz", "_mni0Affine.txt"), "w").write(txt)

    u, s, vt = np.linalg.svd(MI[:3,:3])
    MI3rigid = u @ vt
    txt = """#Insight Transform File V1.0\nTransform: AffineTransform_float_3_3\nFixedParameters: 0 0 0\nParameters: """
    txt += " ".join(["%4.6f %4.6f %4.6f" % tuple(x) for x in MI3rigid.tolist()]) + " %4.6f %4.6f %4.6f\n" % (MI[0,3], MI[1,3], MI[2,3])
    open(outfilename.replace("_tiv.nii.gz", "_mni0Rigid.txt"), "w").write(txt)
        
    # Call antsApplyTransforms    
    command = '"'+os.path.join(scriptpath,'antsApplyTransforms.exe')+'" '
    command +='-i "'+fname+'" '
    command +='-r "'+os.path.join(scriptpath,'hippoboxL_128.nii.gz')+'" '
    command +='-t "'+outfilename.replace("_tiv.nii.gz", "_mni0Affine.txt")+'" '
    command +='-o "'+outfilename.replace("_tiv.nii.gz", "_boxL.nii.gz")+'" '
    command +='--float'
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)   
    (stdout, stderr) = process.communicate()
    if process.returncode!=0:
       print ('WARNING: Subprocess call returned with error')
       print (stderr,stdout); sys.exit (0)     
    
    # Call antsApplyTransforms      
    command = '"'+os.path.join(scriptpath,'antsApplyTransforms.exe')+'" '
    command +='-i "'+fname+'" '
    command +='-r "'+os.path.join(scriptpath,'hippoboxR_128.nii.gz')+'" '
    command +='-t "'+outfilename.replace("_tiv.nii.gz", "_mni0Affine.txt")+'" '
    command +='-o "'+outfilename.replace("_tiv.nii.gz", "_boxR.nii.gz")+'" '
    command +='--float'
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)   
    (stdout, stderr) = process.communicate()
    if process.returncode!=0:
       print ('WARNING: Subprocess call returned with error')         
       print (stderr,stdout); sys.exit (0)     
   

    net = HippoModel2()
    net.load_state_dict(torch.load(os.path.normpath(scriptpath + "/torchparams/params_hipposub2_tall1_00099_00000_train4.pt"), map_location=device))
    net.to(device)
    net.eval()
    
    fnameL=outfilename.replace("_tiv.nii.gz", "_boxL.nii.gz")
    fnameR=outfilename.replace("_tiv.nii.gz", "_boxR.nii.gz")   

    try:
        fn_trans = fnameL.replace("_boxL.nii.gz", "_mni0Affine.txt")
        trans_mat = np.asfarray(open(fn_trans).readlines()[3][12:].split()[:9]).reshape(3,3)
        scale2native = 0.125 * np.abs(det( trans_mat ))
    except:
        print("No transform file found (*_mni0Affine.txt) - can't compute volumes")
        scale2native = 0    

    img = nibabel.load(fnameL)
    assert np.allclose(det(img.affine), -0.125)    
    assert img.shape == (128, 128, 128)

    binput = torch.from_numpy(np.asarray(img.dataobj).astype("float32", copy = False))
    binput -= binput.mean()
    binput /= binput.std()
    with torch.no_grad():
        out1 = net(binput[None,None].to(device)).to("cpu")
        out = np.asarray(out1.argmax(dim=1), np.uint8)[0]

    nibabel.Nifti1Image(out, img.affine).to_filename(fnameL.replace("boxL", "boxL_hippo"))

    if scale2native:
        csv_contentL = ["%d,%s,%4.4f" % (a[0], a[1], b * scale2native)
                            for a, b in zip(code_labels_L, np.bincount(out.ravel()))][1:] \
                     + ["99,L_total,%4.4f" % ((out > 0).sum() * scale2native)]

    if saveprob:
        outclasses = np.rollaxis(np.asarray(torch.softmax(out1[0], dim=0)), 0, 4)
        outclasses[...,0] = 1 - outclasses[...,0]
        i = nibabel.Nifti1Image(np.clip(outclasses, 0.1, 1), img.affine) # remove noise (<.1) and quantify to improve compression
        i.set_data_dtype(np.uint8)
        i.to_filename(fnameL.replace("boxL", "boxL_hippo_prob"))

    img = nibabel.load(fnameR)
    assert np.allclose(det(img.affine), -0.125)  
    assert img.shape == (128, 128, 128)
    binput = torch.from_numpy(np.asarray(img.dataobj)[::-1].astype("float32", copy = True)) # x-flip copy since torch doesn't support it
    binput -= binput.mean()
    binput /= binput.std()
    with torch.no_grad():
        out1 = net(binput[None,None].to(device)).to("cpu")
        out = np.asarray(out1.argmax(dim=1), np.uint8)[0,::-1]
    
    nibabel.Nifti1Image(out, img.affine).to_filename(fnameR.replace("boxR", "boxR_hippo"))
    
    if scale2native:
        csv_contentR = ["%d,%s,%4.4f" % (a[0], a[1], b * scale2native)
                            for a, b in zip(code_labels_R, np.bincount(out.ravel()))][1:] \
                     + ["199,R_total,%4.4f" % ((out > 0).sum() * scale2native)]

    if saveprob:
        outclasses = np.rollaxis(np.asarray(torch.softmax(out1[0], dim=0))[:,::-1], 0, 4)
        outclasses[...,0] = 1 - outclasses[...,0]
        i = nibabel.Nifti1Image(np.clip(outclasses, 0.1, 1), img.affine)
        i.set_data_dtype(np.uint8)
        i.to_filename(fnameR.replace("boxR", "boxR_hippo_prob"))

    if scale2native:
        with open(fnameL.replace("_boxL.nii.gz", "_hipposubvolumes.csv"), "w") as h:
            h.writelines(["code,name,volume\n"] + ["%s\n" % x for x in (csv_contentL + csv_contentR)])
        
    # Call antsApplyTransforms  
    command = '"'+os.path.join(scriptpath,'antsApplyTransforms.exe')+'" '
    command +='-i "'+outfilename.replace("_tiv.nii.gz", "_boxL_hippo.nii.gz")+'" '
    command +='-r "'+fname+'" '
    command +='-t ["'+outfilename.replace("_tiv.nii.gz", "_mni0Affine.txt")+',1]" '
    command +='-o "'+outfilename.replace("_tiv.nii.gz", "_hippoL_native.nii.gz")+'" '
    command +='--float -n MultiLabel[0.1]' 
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)   
    (stdout, stderr) = process.communicate()
    if process.returncode!=0:
       print ('WARNING: Subprocess call returned with error')    
       print (stderr,stdout); sys.exit (0)        

    # Call antsApplyTransforms
    command = '"'+os.path.join(scriptpath,'antsApplyTransforms.exe')+'" '
    command +='-i "'+outfilename.replace("_tiv.nii.gz", "_boxR_hippo.nii.gz")+'" '
    command +='-r "'+fname+'" '
    command +='-t ["'+outfilename.replace("_tiv.nii.gz", "_mni0Affine.txt")+',1]" '
    command +='-o "'+outfilename.replace("_tiv.nii.gz", "_hippoR_native.nii.gz")+'" '
    command +='--float -n MultiLabel[0.1]'
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)   
    (stdout, stderr) = process.communicate()
    if process.returncode!=0:
       print ('WARNING: Subprocess call returned with error')  
       print (stderr,stdout); sys.exit (0)            

    print(" Elapsed time for subject %4.2fs " % (time.time() - Ti))
       
if sys.platform=="win32":
    print("Peak memory used (Gb) " + str(round(psutil.Process().memory_info().peak_wset/ (1024.*1024*1024),2)))
else:
    print("Peak memory used (Gb) " + str(round(resource.getrusage(resource.RUSAGE_SELF)[2] / (1024.*1024),2)))
       
print("Done")

#pause for windows to be able to see messages
if sys.platform=="win32": os.system("pause") # windows

