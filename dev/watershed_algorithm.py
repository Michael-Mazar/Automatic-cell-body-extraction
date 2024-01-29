# %% 
def watershedsegment(thresh,smooth_distance = True,kernel = 3,min_dist=10):
    
    """Uses the watershed segmentation  algorithm to separate cells or nuclei in K intensities
        ----------
        thresh:(N, M) ndarray
            Segmented image
        
        smooth_distance: bool
            If distances will be smoothed
        
        kernel: int 
            Kernel size for smoothing the distances map
        
        Returns
        -------
        labeled_image: (N, M) ndarray
            The label image will have each cell or nuclei labled with a number
        """
    distances = mh.distance(thresh)
   
    if smooth_distance:
        distance = ndimage.gaussian_filter(distances, kernel)
    else:
        distance = distances
    
    
    maxima = feature.peak_local_max(distance, indices=False, exclude_border=False, min_distance=min_dist)
    surface = distance.max() - distance
    spots, t = mh.label(maxima) 
    areas, lines = mh.cwatershed(surface, spots, return_lines=True)
    
    labeled_clusters, num_clusters= mh.label(thresh, np.ones((3,3), bool))
    joined_labels = segmentation.join_segmentations(areas, labeled_clusters)
    labeled_nucl = joined_labels * thresh

    for index, intensity in enumerate(np.unique(labeled_nucl)):
            labeled_nucl[labeled_nucl == intensity] = index   
    return labeled_nucl