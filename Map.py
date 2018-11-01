import cv2
import numpy as np
import Utils

#===========================================
# Map & Frame Data
LastId = 0
GlobalMap = np.empty(0)
GlobalMapId = np.empty(0)
GlobalColor = np.empty(0)
GlobalDes = np.empty(0)
FrameGraph = []
#===========================================
def MappingOneFrame(kp1, kp2, des1, des2, M1, M2, img, cameraMat):
    # Feature points matching
    pts1 = np.asarray([kp1[i].pt for i in range(len(kp1))])
    pts2 = np.asarray([kp2[i].pt for i in range(len(kp2))])
    gkp1, gkp2 = Utils.Arrange2dPoints(pts1, pts2, cameraMat)

    matches, idx1, idx2 = Utils.KnnMatch(des1, des2)
    pts1 = np.asarray([pts1[x.queryIdx] for x in matches])
    pts2 = np.asarray([pts2[x.trainIdx] for x in matches])
  
    # Compute fundamental matrix and filter outlier
    F, mask = cv2.findFundamentalMat(pts1, pts2, method=cv2.RANSAC)
    pts1, pts2 = pts1[mask.ravel() == 1], pts2[mask.ravel() == 1]
    idx1, idx2 = idx1[mask.ravel() == 1], idx2[mask.ravel() == 1]

    # Extract RGB and transform to homogeneous representation
    rgb = np.asarray([img[int(pts2[i, 1]), int(pts2[i, 0]), 0:3] for i in range(pts2.shape[0])]) / 255.
    pts1, pts2 = Utils.Arrange2dPoints(pts1, pts2, cameraMat)

    P, C = RecoverStructure(pts1, pts2, M1, M2, rgb, True)
    N = BuildFrameNode(M2, gkp2, idx2)
    D = des2[idx2]
    
    return P, C, D, N

#===========================================
# Map Handle
def GlobalMapAppend(gmap, gcolor, gdes):
    global GlobalMap
    global GlobalColor
    global GlobalDes
    
    if(GlobalMap.shape[0] == 0):
        GlobalMap = gmap
        GlobalColor = gcolor
        GlobalDes = gdes
    else:
        GlobalMap = np.append(GlobalMap, gmap, axis=0)
        GlobalColor = np.append(GlobalColor, gcolor, axis=0)
        GlobalDes = np.append(GlobalDes, gdes, axis=0)

def BuildFrameNode(M, kp, idx):
    global GlobalMap
    global FrameGraph

    bias = GlobalMap.shape[0]
    fid = -np.ones((len(kp)))
    for i in range(idx.shape[0]):
        fid[idx[i]] = i + bias

    return {"Ext":M, "Kp":kp, "Idx":fid}

def RecoverStructure(pts1, pts2, M1, M2, rgb, color=True):
    P = []
    C = []
    for i in range(pts1.shape[0]):
        p = Utils.TriangulationMultiView([pts1[i], pts2[i]], [M1, M2])
        p = np.asarray([p[0,0], p[1,0], p[2,0]])
        
        if(color == True):
            c = np.asarray([rgb[i,2], rgb[i,1], rgb[i,0]])
        else:
            c = np.asarray([0,0,1])

        P.append(p)
        C.append(c)
    
    return np.asarray(P), np.asarray(C)

def FrameNodeAppend(M, kp, idx):
    node = BuildFrameNode(M, kp, idx)
    FrameGraph.append(node)

def FrameNodeAppend(node):
    FrameGraph.append(node)