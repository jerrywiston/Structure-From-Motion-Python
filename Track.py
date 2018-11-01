import cv2
import numpy as np
import Utils

def CollectLocalMap(gmap, gdes, fnode):
    idx = fnode["Idx"]
    localMap = []
    localDes = []
    for i in range(idx.shape[0]):
        if(idx[i]>=0):
            localMap.append(gmap[int(idx[i])])
            localDes.append(gdes[int(idx[i])])
    
    return np.asarray(localMap), np.asarray(localDes)

def ExtrinsicByPnp(localMap, localDes, currKp, currDes, cameraMat):
    matches, idx1, idx2 = Utils.KnnMatch(localDes, currDes)
    p3ds = np.asarray([localMap[x.queryIdx] for x in matches])
    tkp = np.asarray([currKp[x.trainIdx].pt for x in matches])
    idx1 = np.asarray(idx1)
    idx2 = np.asarray(idx2)

    retval, rvec, tvec, inliers = cv2.solvePnPRansac(p3ds, tkp, cameraMat, None)
    R, _ = cv2.Rodrigues(rvec)
    T = np.array([tvec[0,0], tvec[1,0], tvec[2,0]])  

    return Utils.ExtrinsicMatrix(R,T), idx1, idx2