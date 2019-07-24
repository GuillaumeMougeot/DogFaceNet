"""
DogFaceNet
Bunch of geometric utility functions

Licensed under the MIT License (see LICENSE for details)
Written by Guillaume Mougeot
"""
import numpy as np 

############################################################
#  Image fractioning
############################################################

def frac(h,w,n=5):
    """
    This function takes an image shape as input (h,w) and a integer n. It returns
    the closest image shape (h_out,w_out) of the original one such that h_out < n
    and w_out < n.
    We can then cut the image in h_out*w_out pieces.
    examples:
    frac(500,375,5) returns (4,3)
    frac(333,500,5) returns (2,3)
    """
    fr = float(h)/w
    h_out = 1
    w_out = 1
    if fr > 1:
        for i in range(1,n):
            for j in range(1,i):
                if abs(float(i)/j - fr) < abs(float(h_out)/w_out - fr):
                    h_out = i
                    w_out = j
    else:
        for i in range(1,n):
            for j in range(i,n):
                if abs(float(i)/j - fr) < abs(float(h_out)/w_out - fr):
                    h_out = i
                    w_out = j
    return h_out, w_out


############################################################
#  Area computation
############################################################

def triangle_area(a,b,c):
    """
    Compute area of a triangle given its three summits 2D coordinates
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ab = b-a
    ac = c-a
    return abs(ab[0]*ac[1]-ab[1]*ac[0])/2.0

def quad_area(a,b,c,d):
    return triangle_area(a,b,c) + triangle_area(a,c,d)

def polygone_area(polygone):
    area = 0
    for i in range(1,len(polygone)-1):
        area += triangle_area(polygone[0],polygone[i],polygone[i+1])
    return area


############################################################
#  Intersection computation
############################################################

def vect(u,v):
    """
    "The 2D vectorial product": u ^ v
    """
    return (u[0]*v[1]-u[1]*v[0])

def is_intersect_segment_segment(segment,Segment):
    """
    Check if there is an intersection
    """
    a,b = segment
    A,B = Segment
    aA = A-a
    aB = B-a
    ab = b-a
    
    Aa = a-A
    Ab = b-A
    AB = B-A
    
    return (vect(ab,aA)*vect(ab,aB)<0 and vect(AB,Aa)*vect(AB,Ab)<0)


def intersect_segment_segment(segment,Segment):
    """
    Compute intersection point
    Note: there have to be an intersection, use is_intersect_segment_segment before using this one
    """
    a,b = segment
    A,B = Segment
    Aa = a-A
    AB = B-A
    ab = b-a
    
    M = np.array([AB,-ab]).T
    if np.linalg.det(M)!=0:
        t,_ = np.linalg.inv(M).dot(Aa)
        return A + t*AB


def is_inside_triangle(triangle,point):
    """
    Returns True if the point is in the triangle
    Returns False if triangle is flat
    """
    # It uses the "barycentre" technique
    a,b,c = triangle
    M = np.array([b-a, c-a]).T
    if np.linalg.det(M)==0:
        return False
    else:
        t = np.linalg.inv(M).dot(point-a)
        return (t[0]>0 and t[0]<1 and t[1]>0 and t[1]<1 and t[0]+t[1]<1)

def is_inside_polygone(polygone,point):
    """
    Returns True if the point is in the polygone
    """
    # We divide the polygone in triangles
    current_index = 1
    is_inside = False
    while not is_inside and current_index < len(polygone)-1:
        triangle = np.array([polygone[0],polygone[current_index],polygone[current_index+1]])
        is_inside = is_inside_triangle(triangle,point)
        current_index += 1
    return is_inside

def intersect_polygone_segment(polygone,segment):
    """
    Return the array of intersection points between "polygone" and "segment".
    "polygone" is a convex polygone.
    """
    intersect = np.empty((0,2))
    current_index = 0
    while len(intersect) < 2 and current_index < len(polygone):
        current_segment = np.array([polygone[current_index-1],polygone[current_index]])
        if is_intersect_segment_segment(current_segment,segment):
            intersect = np.vstack((intersect,[intersect_segment_segment(current_segment,segment)]))
        current_index += 1
    return intersect
        


def intersect_polygone_polygone(Polygone,polygone):
    """
    Compute intersection between two convex polygones.
    They are arrays of points, so of shape (?,2) where ? is the length of the array.
    Its returns the array of points of the intersection convex polygone.
    """
    # intersect is the list of:
    #  -points of polygone that are in Polygone,
    #  -points of Polygone that are in polygone,
    #  -intersection point between edges of box and Box
    # This the points of the intersection of two polygones
    
    # For the first polygone
    in_previous_state = is_inside_polygone(Polygone,polygone[-1])
    if in_previous_state:
        intersect = np.array([polygone[-1]])
    else:
        intersect = np.empty((0,2))
        
    for i in range(len(polygone)):
        in_new_state = is_inside_polygone(Polygone,polygone[i])
        
        # List of intersection points between the full Polygone and a segment of polygone
        # Its lenght has to be between 0 and 2 (because of convex constraint)
        intersect_points = intersect_polygone_segment(Polygone,np.array([polygone[i-1],polygone[i]]))
       
        if in_previous_state==True and in_new_state==True: # len(intersect_points)==0
            intersect = np.vstack((intersect, [polygone[i]]))
        elif in_previous_state==True and in_new_state==False: # len(intersect_points)==1
            intersect = np.vstack((intersect, intersect_points))
        elif in_previous_state==False and in_new_state==True: # len(intersect_points)==1
            intersect = np.vstack((intersect, intersect_points))
            intersect = np.vstack((intersect, [polygone[i]]))
        elif len(intersect_points)==2:
            intersect = np.vstack((intersect, intersect_points))
        
        in_previous_state = in_new_state
    
    # For the second polygone
    for i in range(len(Polygone)):
        if is_inside_polygone(polygone,Polygone[i]):
            intersect = np.vstack((intersect, [Polygone[i]]))

    return intersect