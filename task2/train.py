import cv2
import numpy as np
import matplotlib.pyplot as plt

#from google.colab.patches import cv2_imshow
def match_images1(image1, image2):
    # Read images
    img1 = cv2.normalize(image1, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    img2 = cv2.normalize(image2, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

    # Initiate ORB detector
    orb = cv2.ORB_create()

    # Find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)

    # FLANN parameters
    #FLANN_INDEX_KDTREE = 1
    #index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=3)
    #search_params = dict(checks=50)  # or pass empty dictionary

    # Use FLANN to find the best matches
    #flann = cv2.FlannBasedMatcher(index_params, search_params)
    #matches = flann.knnMatch(des1, des2, k=2)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1,des2)

    # Choose matches that passed distance barier

    matc = []
    for i in matches:
      if i.distance < 40:
        matc.append(i)

    matc= sorted(matc, key = lambda x:x.distance)
    #print(matc[0].distance)
    # Draw first 10 matches.
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,matc[:10],None,flags=cv2.DrawMatchesFlags_DEFAULT)
    plt.figure(figsize=(15, 15))
    plt.imshow(img3),plt.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
#match_images1(raster_img1, raster_img2)

def match_images2(image1, image2, max_points=50):
    # Read images
    img1 = cv2.normalize(image1, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    img2 = cv2.normalize(image2, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    if type(des1) != np.uint8:
        des1 = np.asarray(des1, np.uint8)


    if type(des2) != np.uint8 :
        des2 = np.asarray(des2, np.uint8)


    # FLANN parameters
    FLANN_INDEX_LSH = 6
    index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 2) #2
    search_params = dict(checks=50)   # or pass empty dictionary

    # Use FLANN to find the best matches
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for _ in range(len(matches))]

    # ratio test as per Lowe's paper

    for i, (m, n) in enumerate(matches):

        if m.distance < 0.7 * n.distance:
            matchesMask[i] = [1, 0]


    draw_params = dict(matchColor=(0, 255, 0),
                      singlePointColor=(255, 0, 0),
                      matchesMask=matchesMask,
                       flags=0)

    # Draw matches
    img_matches = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)

    # Display the result
    plt.figure(figsize=(15, 15))
    plt.imshow(img_matches),plt.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
#match_images2(raster_img1, raster_img2)
