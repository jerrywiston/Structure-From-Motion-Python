import cv2
import numpy as np

M1 = np.array([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,0]
])

def FeatureCreate():
    return cv2.xfeatures2d.SIFT_create(edgeThreshold = 15)
    #return cv2.ORB_create(5000)

def ExtrinsicMatrix(R, T):
    M = np.zeros((3,4), dtype=np.float32)
    M[0:3,0:3] = R
    M[0:3,3] = T
    return M

def Arrange2dPoints(qkp, tkp, cameraMat):
    z1 = np.ones((qkp.shape[0],1))
    z2 = np.ones((tkp.shape[0],1))
    pts1 = np.append(qkp, z1, axis=1)
    pts2 = np.append(tkp, z2, axis=1)
    pts1 = (np.linalg.inv(cameraMat) * pts1.transpose()).transpose()[:,0:2]
    pts2 = (np.linalg.inv(cameraMat) * pts2.transpose()).transpose()[:,0:2]

    return pts1, pts2

def KnnMatch(des1, des2, dist=0.7):
    bfmatcher = cv2.BFMatcher_create()
    matches = bfmatcher.knnMatch(des1, des2, k=2)

    goodmatches = []
    idx1 = []
    idx2 = []
    max_dist = max([m.distance for m,n in matches])
    for i, (m,n) in enumerate(matches):
        if m.distance < dist*n.distance:
            goodmatches.append(m)
            idx1.append(m.queryIdx)
            idx2.append(m.trainIdx)
    
    idx1 = np.asarray(idx1)
    idx2 = np.asarray(idx2)
    
    return goodmatches, idx1, idx2

def ShowMatch(img1, kp1, img2, kp2, matches, mask, name, scale=4, img_scale=2):
    goodmatches = [m for (m, msk) in zip(matches, mask) if msk == 1]
    result = cv2.drawMatches(img1, kp1, img2, kp2, goodmatches, None)
    cv2.imshow(name, cv2.resize(result, (result.shape[1]*img_scale//scale, result.shape[0]*img_scale//scale)))

def Triangulation(pt1, pt2, M1, M2):
    pt1 = pt1.transpose()
    pt2 = pt2.transpose()
    A = np.matrix([
        [ pt1[0,0]*M1[2,0]-M1[0,0], pt1[0,0]*M1[2,1]-M1[0,1], pt1[0,0]*M1[2,2]-M1[0,2] ],
        [ pt1[1,0]*M1[2,0]-M1[1,0], pt1[1,0]*M1[2,1]-M1[1,1], pt1[1,0]*M1[2,2]-M1[1,2] ],
        [ pt2[0,0]*M2[2,0]-M2[0,0], pt2[0,0]*M2[2,1]-M2[0,1], pt2[0,0]*M2[2,2]-M2[0,2] ],
        [ pt2[1,0]*M2[2,0]-M2[1,0], pt2[1,0]*M2[2,1]-M2[1,1], pt2[1,0]*M2[2,2]-M2[1,2] ]
    ])
    
    B = np.matrix([
        [ -(pt1[0,0]*M1[2,3]-M1[0,3]) ],
        [ -(pt1[1,0]*M1[2,3]-M1[1,3]) ],
        [ -(pt2[0,0]*M2[2,3]-M2[0,3]) ],
        [ -(pt2[1,0]*M2[2,3]-M2[1,3]) ]
    ])

    X = np.linalg.pinv(A)*B
    r = np.matrix([[X[0,0]],[X[1,0]],[X[2,0]],[1]])
    return r

def TriangulationMultiView(pList, mList):
    A = []
    B = []

    for i in range(len(pList)):
        pt = pList[i].transpose()
        M = mList[i]
        A.append([ pt[0,0]*M[2,0]-M[0,0], pt[0,0]*M[2,1]-M[0,1], pt[0,0]*M[2,2]-M[0,2] ])
        A.append([ pt[1,0]*M[2,0]-M[1,0], pt[1,0]*M[2,1]-M[1,1], pt[1,0]*M[2,2]-M[1,2] ])
        B.append([ -(pt[0,0]*M[2,3]-M[0,3]) ])
        B.append([ -(pt[1,0]*M[2,3]-M[1,3]) ])

    A = np.matrix(A)
    B = np.matrix(B)
    X = np.linalg.pinv(A)*B
    r = np.matrix([[X[0,0]],[X[1,0]],[X[2,0]],[1]])
    return r


