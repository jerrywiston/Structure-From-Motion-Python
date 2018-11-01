import cv2
import numpy as np
import Utils
import Map

def ExtractRT(essMat):
    w = np.asarray([[0,-1,0],[1,0,0],[0,0,1]], dtype=np.float32)

    u,s,vt = np.linalg.svd(essMat)
    R1 = np.matmul(np.matmul(u, w), vt)
    R2 = np.matmul(np.matmul(u, np.transpose(w)), vt)
    
    T1 = np.array([u[0,2], u[1,2], u[2,2]])
    T2 = -np.array([u[0,2], u[1,2], u[2,2]])

    return R1, R2, T1, T2

def SelectRT(R1, R2, T1, T2, pts1, pts2, size):
    import random
    score = [0,0,0,0]
    for i in range(size):
        r = random.randint(0,pts1.shape[0]-1)
        for j in range(4):
            if j == 0:
                M2 = Utils.ExtrinsicMatrix(R1, T1)
            elif j == 1:
                M2 = Utils.ExtrinsicMatrix(R2, T1)
            elif j == 2:
                M2 = Utils.ExtrinsicMatrix(R1, T2)
            elif j == 3:
                M2 = Utils.ExtrinsicMatrix(R2, T2)

            p3d1 = Utils.Triangulation(pts1[r], pts2[r], Utils.M1, M2)
            p3d2 = (M2*p3d1)
            if(p3d1[2,0]>0 and p3d2[2,0]>0):
                score[j] += 1

    print(score)
    winner = np.argmax(np.asarray(score))
    if(winner == 0):
        return R1, T1, score[winner]
    elif(winner == 1):
        return R2, T1, score[winner]
    elif(winner == 2):
        return R1, T2, score[winner]
    elif(winner == 3):
        return R2, T2, score[winner]

def InitializeByEssential(img1, img2, cameraMat, sift):
    # SIFT points matching
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
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
    
    # Refine geometric error
    rgb = np.asarray([img1[int(pts1[i, 1]), int(pts1[i, 0]), 0:3] for i in range(pts1.shape[0])]) / 255.
    
    # Extract and find correct RT
    E = cameraMat.T * F * cameraMat
    R1, R2, T1, T2 = ExtractRT(E)
    pts1, pts2 = Utils.Arrange2dPoints(pts1, pts2, cameraMat)
    R, T, sc = SelectRT(R1, R2, T1, T2, pts1, pts2, 100)

    M1 = Utils.M1
    M2 = Utils.ExtrinsicMatrix(R,T)

    # Recover structure by triangulation
    P, C = Map.RecoverStructure(pts1, pts2, Utils.M1, M2, rgb, True)
    N1 = Map.BuildFrameNode(M1, gkp1, idx1)
    N2 = Map.BuildFrameNode(M2, gkp2, idx2)
    D = des2[idx2]

    return M1, M2, P, C, D, N1, N2, kp2, des2