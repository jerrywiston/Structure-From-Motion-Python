import cv2
import numpy as np
from open3d import *

import Utils
import Initialize
import Track
import Map

#===========================================
# Camera matrix

cameraMat = np.asmatrix([
    [2759.48, 0, 1520.69], 
    [0, 2764.16, 1006.81], 
    [0, 0, 1]]
)
'''
cameraMat = np.asmatrix([
    [525.0, 0, 319.5], 
    [0, 525.0, 239.5],
    [0, 0, 1]])
'''
img_scale = 2
cameraMat = cameraMat / img_scale
cameraMat[2,2] = 1

# Output file
def WritePointCloud(filename, pointCloud, rgb, color=False):
    file = open(filename, "w")
    for j in range(pointCloud.shape[0]):
        if color==True:
            file.write("{:.4f}\t{:.4f}\t{:.4f}\t{}\t{}\t{}\n".format(pointCloud[j,0], pointCloud[j,1], pointCloud[j,2], rgb[j,0], rgb[j,1], rgb[j,2]))
        else:
            file.write("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(pointCloud[j,0], pointCloud[j,1], pointCloud[j,2], 0, 0, 1))

#===========================================
# Main
DATA_PATH = "Images/"
DATA_TYPE = ".jpg"

sift = Utils.FeatureCreate()
if __name__ == '__main__':
    # Get image
    ii = 0
    img = []
    for i in range(25-ii):
        img_temp = cv2.imread(DATA_PATH + str(ii+i).zfill(4) + DATA_TYPE, 1)
        img_temp = cv2.resize(img_temp, (img_temp.shape[1]//img_scale, img_temp.shape[0]//img_scale))
        img.append(img_temp)

    # Initialization 
    M1, M2, P, C, D, N1, N2, kp_rec, des_rec = Initialize.InitializeByEssential(img[0], img[1], cameraMat, sift)
    Map.FrameNodeAppend(N1)
    Map.FrameNodeAppend(N2)
    Map.GlobalMapAppend(P, C, D)
    print(M2)
    print(Map.GlobalMap.shape)
    print()
    
    # Start tracking
    M_rec = M2
    for i in range(2,len(img)):
        print("[Frame " + str(i) + "]")
        scale = 1
        cv2.imshow("View", cv2.resize(img[i], (img[i].shape[1]//scale, img[i].shape[0]//scale)))
        #localMap, localDes = Track.CollectLocalMap(Map.GlobalMap, Map.GlobalDes, Map.FrameGraph[i-1])
        #M = Track.ExtrinsicByPnp(localMap, localDes, img[i], cameraMat)
        kp, des = sift.detectAndCompute(img[i],None)
        M, idx1, idx2 = Track.ExtrinsicByPnp(Map.GlobalMap, Map.GlobalDes, kp, des, cameraMat)
        P, C, D, N = Map.MappingOneFrame(kp_rec, kp, des_rec, des, M_rec, M, img[i], cameraMat)
        
        # Record data
        M_rec = M
        kp_rec = kp
        des_rec = des
        
        # Add global map
        Map.FrameNodeAppend(N)
        Map.GlobalMapAppend(P, C, D)
        print(M)
        print(Map.GlobalMap.shape)
        print()
        cv2.waitKey(1)
    
    # Write file & 3D Visualization
    print("Press [Enter] to visualize pointclouds ...")
    WritePointCloud("test.xyzrgb", Map.GlobalMap, Map.GlobalColor, True)
    cv2.waitKey(0)
    pcd = read_point_cloud("test.xyzrgb")
    mesh_frame = create_mesh_coordinate_frame(size = 0.3, origin = [0,0,0])
    draw_geometries([pcd, mesh_frame])
