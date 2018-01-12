from skimage import io
import numpy as np
import os, sys

def eigenface(X, reconstruct_img, size=(600, 600, 3)):
    X = X.reshape(415,600*600*3)
    mean = np.mean(X,axis=0)
    X_center = X - mean
    U, S, V = np.linalg.svd(X_center.T, full_matrices=False)
    
    reconstruct_img = reconstruct_img.reshape(1,600*600*3)
    reconstruct_img = reconstruct_img - mean

    W = np.dot(reconstruct_img, U)
    reconstruct = (mean + np.dot(W[0, 0:4], U[:, 0:4].T))
    reconstruct -= np.min(reconstruct)
    reconstruct = reconstruct/np.max(reconstruct)
    reconstruct = (reconstruct*255).astype(np.uint8)
     
    io.imsave('reconstruction.jpg', reconstruct.reshape(size))

def main():
    dirname = sys.argv[1]
    reconstruct_img = sys.argv[2]
    files = sorted(os.listdir(dirname), key=lambda item: (len(item), item))
    faces = []
    reconstruct_img = io.imread(dirname+'/'+reconstruct_img).reshape(360000,3)
    for imgfile in files:
        face = io.imread(dirname+'/'+imgfile).reshape(360000,3)
        faces.append(face)
    faces = np.array(faces)
    eigenface(faces, reconstruct_img)

if __name__ == "__main__":
    main()
