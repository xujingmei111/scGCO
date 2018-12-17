import math
import random
import parmap
import itertools
from itertools import repeat
from scipy.spatial import distance
import operator
from tqdm import tqdm
from functools import reduce
from sklearn import mixture
import statsmodels.stats.multitest as multi
import networkx as nx
import multiprocessing as mp
import numpy as np
from scipy.stats import poisson
import pandas as pd
import pygco as pygco # cut_from_graph # pip install git+git://github.com/amueller/gco_python
import scipy.stats.mstats as ms
from itertools import repeat
from scipy.stats import norm
from scipy.stats import binom
from scipy.sparse import issparse
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.path as mplPath
import shapely.geometry
import shapely.ops
from scipy.spatial import Voronoi, voronoi_plot_2d, Delaunay, KDTree, ConvexHull
from matplotlib.backends.backend_pdf import PdfPages
import sklearn.manifold as manifold
import sklearn.decomposition as decomposition 
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde
from sklearn.cluster import DBSCAN

def create_graph_with_weight(points, normCount):
    '''
    Returns a graph created from cell coordiantes.
    edge weights set by normalized counts.
    
    :param points: shape (n,2); normCount: shape (n)
    :rtype: ndarray shape (n ,3)
    
    '''
    edges = {}   
    var = normCount.var()
    delauny = Delaunay(points)
#    cellGraph = np.zeros((delauny.simplices.shape[0]*delauny.simplices.shape[1], 4))
    cellGraph = np.zeros((points.shape[0]*10, 4))

    for simplex in delauny.simplices:
        simplex.sort()
        edge0 = str(simplex[0]) + " " + str(simplex[1])
        edge1 = str(simplex[0]) + " " + str(simplex[2])
        edge2 = str(simplex[1]) + " " + str(simplex[2])
        edges[edge0] = 1
        edges[edge1] = 1
        edges[edge2] = 1
        
    i = 0
    for kk in edges.keys():  
        node0 = int(kk.split(sep=" ")[0])
        node1 = int(kk.split(sep=" ")[1])
        edgeDiff = normCount[node0] - normCount[node1]
        energy = np.exp((0 - edgeDiff**2)/(2*var))
        dist = distance.euclidean(points[node0,:], points[node1,:])
        cellGraph[i] = [node0, node1, energy, dist]       
        i = i + 1
        
    tempGraph = cellGraph[0:i]
    n_components_range = range(1,5)
    best_component = 1
    lowest_bic=np.infty
    temp_data = tempGraph[:,3].reshape(-1,1)
    for n_components in n_components_range:
        gmm = mixture.GaussianMixture(n_components = n_components)
        gmm.fit(temp_data)
        gmm_bic = gmm.bic(temp_data)
        if gmm_bic < lowest_bic:
            best_gmm = gmm
            lowest_bic = gmm_bic
            best_component = n_components  
    
    mIndex = np.where(best_gmm.weights_ == max(best_gmm.weights_))[0]
    cutoff = best_gmm.means_[mIndex] + 2*np.sqrt(best_gmm.covariances_[mIndex])

    for simplex in delauny.simplices:
        simplex.sort()          
        dist0 = distance.euclidean(points[simplex[0],:], points[simplex[1],:])
        dist1 = distance.euclidean(points[simplex[0],:], points[simplex[2],:])
        dist2 = distance.euclidean(points[simplex[1],:], points[simplex[2],:])
        tempArray = np.array((dist0, dist1, dist2))
        badIndex = np.where(tempArray == max(tempArray))[0][0]
        if tempArray[badIndex] > cutoff:
            edge0 = str(simplex[0]) + " " + str(simplex[1])  
            edge1 = str(simplex[0]) + " " + str(simplex[2])       
            edge2 = str(simplex[1]) + " " + str(simplex[2])
            edgeCount = 0
            if edge0 in edges and edge1 in edges and edge2 in edges:
                if badIndex == 0:
                    del edges[edge0]
                elif badIndex == 1:
                    del edges[edge1]
                elif badIndex == 2:
                    del edges[edge2]     

    i = 0
    for kk in edges.keys():  
        node0 = int(kk.split(sep=" ")[0])
        node1 = int(kk.split(sep=" ")[1])
        edgeDiff = normCount[node0] - normCount[node1]
        energy = np.exp((0 - edgeDiff**2)/(2*var))
        dist = distance.euclidean(points[node0,:], points[node1,:])
        cellGraph[i] = [node0, node1, energy, dist]       
        i = i + 1   
        
    tempGraph = cellGraph[0:i]
    temp_data = tempGraph[:,3].reshape(-1,1)    
    gmm = mixture.GaussianMixture(n_components = 1)
    gmm.fit(temp_data)    
    cutoff = gmm.means_[0] + 2*np.sqrt(gmm.covariances_[0])
    
    finalGraph = tempGraph.copy()
    j=0
    for i in np.arange(tempGraph.shape[0]):    
        if tempGraph[i, 3] < cutoff:
            finalGraph[j] = tempGraph[i]
            j = j + 1
            
    return finalGraph[0:j]  


def read_spatial_expression(file):
    '''
    Returns pandas data frame of spatial gene express
    and numpy ndarray for single cell location coordinates.
    
    :param file: csv file for spatial gene expression; 
    :rtype: coord (spatial coordinates) shape (n, 2); data: shape (n, m); 
    '''
    data = pd.read_csv(file, sep='\s+', index_col = 0)
    temp = [val.split('x') for val in data.index.values]
    coord = np.array([[float(a[0]), float(a[1])] for a in temp])
    return coord, data


def rotate(origin, point, angle):
    """
    Rotate a point around another points (origin).
    
    :param file: coordiantes of points; angle in radians 
    :rtype: spatial coordinates; 

    """
    ox, oy = origin
    px, py = point

    dx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    dy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return dx, dy


def plot_voronoi(geneID, coord, count, 
                 line_colors = 'k', line_width = 0.5, line_alpha = 1.0):
    '''
    plot spatial expression as voronoi tessellation
    
    :param file: geneID; spatial coordinates coord: shape (n, 2); normalized count: shape (n); 
                line_colors = 'k'; line_width = 0.5; line_alpha = 1.0
    '''
    points = coord
    tempPoints = points
    hull = ConvexHull(points)
    polygon = shapely.geometry.Polygon(points[hull.vertices])
    for simplex in hull.simplices:
        point = rotate(points[simplex[0]], points[simplex[1]], math.radians(60))
        if polygon.contains(shapely.geometry.Point(point)):
            point = rotate(points[simplex[0]], points[simplex[1]], math.radians(-60))
        np.append(tempPoints, [point], axis = 0)
    vorTemp = Voronoi(tempPoints)
    vor = Voronoi(points) 
    minima = min(count)
    maxima = max(count)
    norm = mpl.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.PiYG)
    mapper.set_array(count)
    voronoi_plot_2d(vor, show_points=True, show_vertices=False, 
                    line_colors = line_colors, line_width = line_width, 
                    line_alpha = line_alpha)
    for r in range(len(vor.point_region)):
        region = vor.regions[vor.point_region[r]]
        if not -1 in region:
            polygon = [vor.vertices[i] for i in region]
            plt.fill(*zip(*polygon), color=mapper.to_rgba(count[r]))

    plt.colorbar(mapper)
    plt.title(geneID)
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.show()

def pdf_voronoi(geneID, coord, count, fileName, 
                line_colors = 'k', line_width = 0.5, line_alpha = 1.0):
    '''
    save spatial expression as voronoi tessellation to pdf
    
    :param file: geneID; spatial coordinates shape (n, 2); normalized count: shape (n); 
                fileName; line_colors = 'k'; line_width = 0.5; line_alpha = 1.0
    
    '''
    points = coord
    tempPoints = points
    hull = ConvexHull(points)
    polygon = shapely.geometry.Polygon(points[hull.vertices])
    for simplex in hull.simplices:
        point = rotate(points[simplex[0]], points[simplex[1]], math.radians(60))
        if polygon.contains(shapely.geometry.Point(point)):
            point = rotate(points[simplex[0]], points[simplex[1]], math.radians(-60))
        np.append(tempPoints, [point], axis = 0)
    vorTemp = Voronoi(tempPoints)
    vor = Voronoi(points) 
    minima = min(count)
    maxima = max(count)
    norm = mpl.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.PiYG)
    mapper.set_array(count)
    voronoi_plot_2d(vor, show_points=True, show_vertices=False, line_colors = line_colors, line_width = line_width, line_alpha = line_alpha)
    for r in range(len(vor.point_region)):
        region = vor.regions[vor.point_region[r]]
        if not -1 in region:
            polygon = [vor.vertices[i] for i in region]
            plt.fill(*zip(*polygon), color=mapper.to_rgba(count[r]))

    plt.colorbar(mapper)
    plt.title(geneID)
    if fileName != None:
        plt.savefig(fileName)
    else:
        print('ERROR! Please supply a file name.')

    
    
def plot_voronoi_boundary(geneID, coord, count, 
                          classLabel, p, fdr=False,  
                          line_colors = 'k', class_line_width = 3, 
                          line_width = 0.5, line_alpha = 1.0):
    '''
    plot spatial expression as voronoi tessellation
    highlight boundary between classes
    
    :param file: geneID; spatial coordinates shape (n, 2); normalized count: shape (n);
                predicted cell class calls shape (n); prediction p-value.
                fdr=False; line_colors = 'k'; class_line_width = 3; 
                line_width = 0.5; line_alpha = 1.0
    '''
    points = coord
    count = count
    labels = classLabel
    vor = Voronoi(points)

    minima = min(count)
    maxima = max(count)
    norm = mpl.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.PiYG)
    mapper.set_array(count)

    voronoi_plot_2d(vor, show_points=True, show_vertices=False, line_colors = line_colors, line_width = line_width, line_alpha = line_alpha)
    for r in range(len(vor.point_region)):
        region = vor.regions[vor.point_region[r]]
        if not -1 in region:
            polygon = [vor.vertices[i] for i in region]
            plt.fill(*zip(*polygon), color=mapper.to_rgba(count[r]))
   
    # plot ridge between two points 
    center = points.mean(axis=0)
    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = np.asarray(simplex)
#        print(pointidx)
        # check whether the two points belong to different classes
        # and plot with color accoding to the classes
        if np.any(simplex < 0):
            if labels[pointidx[0]] != labels[pointidx[1]]:
                i = simplex[simplex >= 0][0] # finite end Voronoi vertex
                t = points[pointidx[1]] - points[pointidx[0]]  # tangent
                t = t / np.linalg.norm(t)
                n = np.array([-t[1], t[0]]) # normal
                midpoint = points[pointidx].mean(axis=0)
                far_point = vor.vertices[i] + np.sign(np.dot(midpoint - center, n)) * n * 100
                plt.plot([vor.vertices[i,0], far_point[0]],[vor.vertices[i,1], far_point[1]], 'k--')
        else:
            if labels[pointidx[0]] != labels[pointidx[1]]:
                plt.plot(vor.vertices[simplex, 0], vor.vertices[simplex, 1], 'k-', lw=class_line_width)
                
    plt.xlim(min(points[:,0])-0.5, max(points[:,0]) + 0.5); 
    plt.ylim(min(points[:,1])-0.5, max(points[:,1]) + 0.5)
    plt.colorbar(mapper)
    if fdr:
        titleText = geneID + '\n' + 'fdr: ' + str("{:.2e}".format(p))
    else:
        titleText = geneID + '\n' + 'p_value: ' + str("{:.2e}".format(p))    
    plt.title(titleText)
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.show()

    
def pdf_voronoi_boundary(geneID, coord, count, 
                          classLabel, p, fileName, fdr=False, 
                          line_colors = 'k', class_line_width = 3, 
                          line_width = 0.5, line_alpha = 1.0):
    '''
    save spatial expression as voronoi tessellation to pdf
    highlight boundary between classes
    
    :param file: geneID; spatial coordinates shape (n, 2); normalized count: shape (n);
                predicted cell class calls shape (n); prediction p-value; pdf fileName;
                fdr=False; line_colors = 'k'; class_line_width = 3; 
                line_width = 0.5; line_alpha = 1.0
    '''
    points = coord
    count = count
    labels = classLabel
    vor = Voronoi(points)

    minima = min(count)
    maxima = max(count)
    norm = mpl.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.PiYG)
    mapper.set_array(count)

    voronoi_plot_2d(vor, show_points=True, show_vertices=False, line_colors = line_colors, line_width = line_width, line_alpha = line_alpha)
    for r in range(len(vor.point_region)):
        region = vor.regions[vor.point_region[r]]
        if not -1 in region:
            polygon = [vor.vertices[i] for i in region]
            plt.fill(*zip(*polygon), color=mapper.to_rgba(count[r]))
   
    # plot ridge between two points 
    center = points.mean(axis=0)
    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = np.asarray(simplex)
#        print(pointidx)
        # check whether the two points belong to different classes
        # and plot with color accoding to the classes
        if np.any(simplex < 0):
            if labels[pointidx[0]] != labels[pointidx[1]]:
                i = simplex[simplex >= 0][0] # finite end Voronoi vertex
                t = points[pointidx[1]] - points[pointidx[0]]  # tangent
                t = t / np.linalg.norm(t)
                n = np.array([-t[1], t[0]]) # normal
                midpoint = points[pointidx].mean(axis=0)
                far_point = vor.vertices[i] + np.sign(np.dot(midpoint - center, n)) * n * 100
                plt.plot([vor.vertices[i,0], far_point[0]],[vor.vertices[i,1], far_point[1]], 'k--')
        else:
            if labels[pointidx[0]] != labels[pointidx[1]]:
                plt.plot(vor.vertices[simplex, 0], vor.vertices[simplex, 1], 'k-', lw=class_line_width)
                
    plt.xlim(min(points[:,0])-0.5, max(points[:,0]) + 0.5); 
    plt.ylim(min(points[:,1])-0.5, max(points[:,1]) + 0.5)
    plt.colorbar(mapper)
    if fdr:
        titleText = geneID + '\n' + 'fdr: ' + str("{:.2e}".format(p))
    else:
        titleText = geneID + '\n' + 'p_value: ' + str("{:.2e}".format(p))    
    plt.title(titleText)
    plt.axis('off')
#    plt.xlabel('X coordinate')
#    plt.ylabel('Y coordinate')
    if fileName != None:
        plt.savefig(fileName)
    else:
        print('ERROR! Please supply a file name.')


def find_mixture(data):
    '''
    estimate expression clusters
    
    :param file: data (n,); 
    :rtype: gmm object;
    '''
    n_components_range = range(2,5)
    best_component = 2
    lowest_bic=np.infty
    temp_data = data.reshape(-1,1)
    for n_components in n_components_range:
        gmm = mixture.GaussianMixture(n_components = n_components)
        gmm.fit(temp_data)
        gmm_bic = gmm.bic(temp_data)
        if gmm_bic < lowest_bic:
            best_gmm = gmm
            lowest_bic = gmm_bic
            best_component = n_components      

    return best_gmm
# np.percentile(s, 10)

def find_mixture_2(data):
    '''
    estimate expression clusters, use k=2
    
    :param file: data (n,); 
    :rtype: gmm object;
    '''
    gmm = mixture.GaussianMixture(n_components = 2)
    gmm.fit(data.reshape(-1,1))

    return gmm

def normalize_count_cellranger(data):
    '''
    normalize count as in cellranger
    
    :param file: data: A dataframe of shape (m, n);
    :rtype: data shape (m, n);
    '''
    normalizing_factor = np.sum(data, axis = 1)/np.median(np.sum(data, axis = 1))
    data = pd.DataFrame(data.values/normalizing_factor[:,np.newaxis], columns=data.columns, index=data.index)
    return data


def get_gene_high_dispersion(data, n_bins=20, z_cutoff=1.7, min_mean=0.0125, max_mean=3, min_disp=0.5):
    '''
    identify genes with high dispersion as in cellranger
    
    :param file: data (m,n); n_bins=20; z_cutoff=1.7; min_mean=0.0125; max_mean=3; min_disp=0.5.
    :rtype: ndarray (k, ); only index of genes with significant variations are returned
    '''
    mean = np.mean(data, axis=0)
    var = np.var(data, axis=0)
    mean[mean == 0] = 1e-12
    dispersion = var/mean
    temp = np.vstack((np.log1p(mean), np.log(dispersion))).T
    vmd = pd.DataFrame(temp, index=data.columns, columns=['mean', 'disp'])
    
    vmd['bins'] = pd.cut(vmd['mean'], bins=n_bins)
    disp_binned = vmd.groupby('bins')['disp']
    disp_binned_mean = disp_binned.mean()
    disp_binned_std = disp_binned.std()
    disp_binned_mean = disp_binned_mean[vmd['bins']].values
    disp_binned_std = disp_binned_std[vmd['bins']].values
    vmd['disp_norm'] = (vmd['disp'].values - disp_binned_mean) / disp_binned_std
    good_std = ~np.isnan(disp_binned_std)    
    
    mean_sel = np.logical_and(vmd['mean'] > min_mean, vmd['mean'] < max_mean)
    disp_sel = np.logical_and(vmd['disp'] > min_disp, vmd['disp_norm'] > z_cutoff)
    sel_1 = np.logical_and(mean_sel, disp_sel)
    sel = np.logical_and(sel_1, good_std)    
    return np.argwhere(sel).flatten()

def log1p(data):
    '''
    log transform normalized count data
    
    :param file: data (m, n); 
    :rtype: data (m, n);
    '''
    if not issparse(data):
        return np.log1p(data)
    else:
        return data.log1p()

def first_neg_index(a):
    '''
    find the first negative index
    :param a shape(n, )
    :rtype scalar
    '''
    for i in np.arange(a.shape[0]):
        if a[i] < 0:
            return i
    return a.shape[0] - 1
    
    
def calc_u_cost(a, mid_points):
    '''
    calculate unary energy
    :param: a shape(n, ); mid_points scalar
    :rtype: scalar
    '''
    neg_index = int(a[0])
    x = a[1]
    m_arr = np.concatenate((0 - mid_points[0:neg_index+1], 
                            mid_points[neg_index:]), axis=0)
    x_arr = np.concatenate((np.repeat(x, neg_index+1), 
                0 - np.repeat(x, mid_points.shape[0] - neg_index)), axis=0)
    return m_arr+x_arr


def cut_graph_potts(cellGraph, count, potts_k = 100, unary_scale_factor=100, 
                    smooth_factor=100, label_cost=10, algorithm='expansion'):
    '''
    Returns new labels and energy for the cut.
    
    :param points: cellGraph shape (n,3); count: shape (n,); 
    :unary_scale_factor, scalar; smooth_factor, scalar; 
    :label_cost: scalar; potts_k: scalar.
    :rtype: label shape (n,); gmm object.
    '''
    a = count.copy()
    a = a[a > 0]
    gmm = find_mixture(a)
    if gmm.means_.shape[0] == 1:
        gmm = find_mixture_2(a)
    unary_cost = compute_unary_cost_potts(count, gmm, unary_scale_factor)
#    unary_cost = (unary_cost-np.median(unary_cost))
#    unary_cost = unary_cost*1000/np.abs(unary_cost).max()
    pairwise_cost = compute_pairwise_cost(gmm.means_.shape[0], smooth_factor)
    edges = cellGraph[:,0:2].astype(np.int32)
#    edge_weights = cellGraph[:,2:3]
    labels = pygco.cut_from_graph(edges, unary_cost, pairwise_cost, label_cost)
    return labels, gmm

def cut_graph_general(cellGraph, count, unary_scale_factor=100, 
                      smooth_factor=50, label_cost=10, algorithm='expansion'):
    '''
    Returns new labels and gmm for the cut.
    
    :param points: cellGraph (n,3); count: shape (n,); 
    :unary_scale_factor, scalar; smooth_factor, scalar; 
    :label_cost: scalar; algorithm='expansion'
    :rtype: label shape (n,); gmm object.
    '''
    a = count.copy()
    a = a[a > 0]
    gmm = find_mixture(a)
    unary_cost = compute_unary_cost_simple(count, gmm, unary_scale_factor)
    
    pairwise_cost = compute_pairwise_cost(gmm.means_.shape[0], smooth_factor)
    edges = cellGraph[:,0:2].astype(np.int32)
    labels = pygco.cut_from_graph(edges, unary_cost, pairwise_cost, label_cost)
#    energy = compute_energy(unary_cost, pairwise_cost, edges, labels)

    return labels, gmm 


def compute_unary_cost_simple(count, gmm, scale_factor):
    '''
    Returns unary cost energy.
    
    :param points: count: shape (n,); gmm: gmm object; scale_factor: scalar

    :rtype: unary energy matrix.
    '''    
    labels_pred = gmm.predict(count.reshape(-1,1))
    temp_means = np.sort(gmm.means_, axis=None)
    new_index = np.where(gmm.means_ == temp_means)[1]
    temp_covs = gmm.covariances_.copy()
    for i in np.arange(new_index.shape[0]):
        temp_covs[i] = gmm.covariances_[new_index[i]]
    new_labels = np.zeros(labels_pred.shape[0], dtype=np.int32)
    for i in np.arange(new_index.shape[0]):
        temp_index = np.where(labels_pred == i)[0]
        new_labels[temp_index] = new_index[i]

    mid_points = np.zeros(len(new_index) - 1)
    for i in np.arange(len(mid_points)):
        mid_points[i] = (temp_means[i]*np.sqrt(temp_covs[i+1]) + 
                     temp_means[i+1]*np.sqrt(temp_covs[i])
                    )/(np.sqrt(temp_covs[i]) + np.sqrt(temp_covs[i+1]))
    temp = count[:, np.newaxis] - temp_means.T[1:]
    neg_indices = np.apply_along_axis(first_neg_index, 1, temp)
    ind_count_arr = np.vstack((neg_indices, count)).T        
    return (scale_factor*np.apply_along_axis(calc_u_cost, 1, 
                                    ind_count_arr, mid_points)).astype(np.int32)


def compute_pairwise_cost(size, smooth_factor):
    '''
    Returns pairwise energy.
    
    :param points: size: scalar; smooth_factor: scalar

    :rtype: pairwise energy matrix.
    '''
    pairwise_size = size
    pairwise = -smooth_factor * np.eye(pairwise_size, dtype=np.int32)
    step_weight = -smooth_factor*np.arange(pairwise_size)[::-1]
    for i in range(pairwise_size): 
        pairwise[i,:] += np.roll(step_weight,i) 
    temp = np.triu(pairwise).T + np.triu(pairwise)
    np.fill_diagonal(temp, np.diag(temp)/2)
    return temp
                       
    
def compute_energy(unary_cost, pairwise_cost, edges, labels):
    '''
    Returns energy of the cut.
    
    :param points: unary_cost: shape (n,n); pairwise_cost: shape(k,k)
                    edges: ; labels: shape(n,)

    :rtype: unary energy matrix.
    '''
    temp_energy = np.zeros(unary_cost.shape)
    lab_index = np.zeros((unary_cost.shape[0], 2)).astype(np.int32)
    lab_index[:, 0] = np.arange(unary_cost.shape[0]).astype(np.int32)
    lab_index[:, 1] = labels
    temp_energy[lab_index[:,0], lab_index[:,1]] = 1
    return np.sum(np.multiply(temp_energy, unary_cost))

def remove_egdes(edges, newLabels):
    '''
    Mark boundary of the cut.
    
    :param points: edges: shape (n,); newLabels: shape(k,)

    :rtype: marked edges.
    '''
    if newLabels[int(edges[0])] != newLabels[int(edges[1])]:
        edges[2] = 0
    else:
        edges[2] = 1
    return edges

def remove_single_link(G):
    '''
    parse graph to remove component with single link
    :param G: networkX graph object
    :rtype: networkX graph object
    '''
    # remove longest strectch with single deg nodes first
    # remove connected 4 node with degree 2
    node_list = list(set(G.nodes))
    node_deg1 = np.array(G.degree())[
            np.where(np.array(G.degree())[:,1]==1)[0],0]    
    subG2 = G.subgraph(np.array(G.degree())[
            np.where(np.array(G.degree())[:,1]<=2)[0], 0])
    con_components2 = sorted(nx.connected_components(subG2), 
                                  key = len, reverse=True)   
    for con in con_components2:
        if (len(set(con).intersection(set(node_deg1))) > 0):
            node_list= set(node_list).difference(set(con))   
    subG = G.subgraph(list(node_list))
    subG2 = subG.subgraph(np.array(subG.degree())[
            np.where(np.array(subG.degree())[:,1]<=2)[0], 0])
    con_components2 = sorted(nx.connected_components(subG2), 
                                  key = len, reverse=True)       
    for con in con_components2:
        if (len(con) > 3):
            node_list= set(node_list).difference(set(con))
  # round 2  
    subG = G.subgraph(list(node_list))
    node_deg1 = np.array(subG.degree())[
            np.where(np.array(subG.degree())[:,1]==1)[0],0]   
    if len(node_deg1) > 0:
        subG2 = subG.subgraph(np.array(subG.degree())[
            np.where(np.array(subG.degree())[:,1]<=2)[0], 0])
        con_components2 = sorted(nx.connected_components(subG2), 
                                  key = len, reverse=True)   
        for con in con_components2:
            if (len(set(con).intersection(set(node_deg1))) > 0):
                node_list= set(node_list).difference(set(con))   
        subG = G.subgraph(list(node_list))
#        print(subG.degree())
#        print(np.where(np.reshape(np.array(subG.degree()), (-1,2))[:,1]<=2))
        subG2 = subG.subgraph(np.array(subG.degree())[
            np.where(np.array(subG.degree())[:,1]<=2)[0], 0])
        con_components2 = sorted(nx.connected_components(subG2), 
                                  key = len, reverse=True)       
        for con in con_components2:
            if (len(con) > 3):
                node_list= set(node_list).difference(set(con))    

  # round 3  
    subG = G.subgraph(list(node_list))
    node_deg1 = np.reshape(np.array(subG.degree()), (-1,2))[
            np.where(np.reshape(np.array(subG.degree()), (-1,2))[:,1]==1)[0],0]   
    if len(node_deg1) > 0:
        subG2 = subG.subgraph(np.array(subG.degree())[
            np.where(np.reshape(np.array(subG.degree()), (-1,2))[:,1]<=2)[0], 0])
        con_components2 = sorted(nx.connected_components(subG2), 
                                  key = len, reverse=True)   
        for con in con_components2:
            if (len(set(con).intersection(set(node_deg1))) > 0):
                node_list= set(node_list).difference(set(con))   
        subG = G.subgraph(list(node_list))
        subG2 = subG.subgraph(np.reshape(np.array(subG.degree()), (-1,2))[
            np.where(np.reshape(np.array(subG.degree()), (-1,2))[:,1]<=2)[0], 0])
        con_components2 = sorted(nx.connected_components(subG2), 
                                  key = len, reverse=True)       
        for con in con_components2:
            if (len(con) > 3):
                node_list= set(node_list).difference(set(con))                   
                
    subG = G.subgraph(list(node_list))
        
    return subG

def compute_p_CSR(newLabels, gmm, exp, cellGraph): 
    '''
    Returns p_value of the cut.
    
    :param points: newLabels: shape (n,); gmm: gmm object
                   exp: shape (n ,); cellGraph: shape (n,3)

    :rtype: p_value.
    '''
    com_factor = 9
    p_values = list()
    node_lists = list()
    gmm_pred = gmm.predict(exp.reshape(-1,1))
    unique, counts = np.unique(gmm_pred,return_counts=True)
    G=nx.Graph()
    tempGraph = cellGraph.copy()
    tempGraph = np.apply_along_axis(remove_egdes, 1, tempGraph, newLabels)
    G.add_edges_from(tempGraph[np.where(tempGraph[:,2] == 1)[0],0:2].astype(np.int32))
    con_components_old = sorted(nx.connected_components(G), 
                                  key = len, reverse=True)   
    G = remove_single_link(G)
    con_components = sorted(nx.connected_components(G), 
                                  key = len, reverse=True)

    for j in np.arange(len(con_components)):
        node_list = con_components[j]
        com_size = len(node_list)
        if com_size >= com_factor:
            gmm_pred_com = gmm.predict(exp[np.array(list(node_list))].reshape(-1,1))
            unique_com, counts_com = np.unique(gmm_pred_com, return_counts=True)
            major_label = unique_com[np.where(counts_com == counts_com.max())[0][0]]
            label_count = counts[np.where(unique == major_label)[0]]
            count_in_com =  counts_com.max()
            prob = poisson.sf(count_in_com, com_size*(label_count/exp.shape[0]))[0]

            p1 = poisson.pmf(count_in_com, com_size*(label_count/exp.shape[0]))[0]
 #           print(prob, count_in_com, com_size, com_size*(label_count/exp.shape[0]))
            p = prob + p1
            p_values.append(p)
            node_lists.append(np.array(list(node_list)))
    return p_values, node_lists, con_components_old


def compute_p_CSR_simple(newLabels, gmm, exp, cellGraph): 
    '''
    Returns p_value of the cut.
    
    :param points: newLabels: shape (n,); gmm: gmm object
                   exp: shape (n ,); cellGraph: shape (n,)

    :rtype: p_value.
    '''
    p_values = list()
    node_lists = list()
    gmm_pred = gmm.predict(exp.reshape(-1,1))
    com_factor = 9
    unique, counts = np.unique(gmm_pred,return_counts=True)
    G=nx.Graph()
    tempGraph = cellGraph.copy()
    tempGraph = np.apply_along_axis(remove_egdes, 1, tempGraph, newLabels)
    G.add_edges_from(tempGraph[np.where(tempGraph[:,2] == 1)[0],0:2].astype(np.int32))
    con_components_old = sorted(nx.connected_components(G), 
                                  key = len, reverse=True)   
#    G = remove_single_link(G)
    con_components = sorted(nx.connected_components(G), 
                                  key = len, reverse=True)

    for j in np.arange(len(con_components)):
        node_list = con_components[j]
        com_size = len(node_list)
        if com_size >= com_factor:
            gmm_pred_com = gmm.predict(exp[np.array(list(node_list))].reshape(-1,1))
            unique_com, counts_com = np.unique(gmm_pred_com, return_counts=True)
            major_label = unique_com[np.where(counts_com == counts_com.max())[0][0]]
            label_count = counts[np.where(unique == major_label)[0]]
            count_in_com =  counts_com.max()
            prob = poisson.sf(count_in_com, com_size*(label_count/exp.shape[0]))[0]

            p1 = poisson.pmf(count_in_com, com_size*(label_count/exp.shape[0]))[0]
 #           print(prob, count_in_com, com_size, com_size*(label_count/exp.shape[0]))
            p = prob + p1
            p_values.append(p)
            node_lists.append(np.array(list(node_list)))
    return p_values, node_lists, con_components_old

def compute_alpha_shape(points):
    '''
    Returns list of abnormal cells (a pruning for Delaunay tessellation).
    
    :param points: ndarray shape (n, 2); 

    :rtype: shape (n, ). 
    
    '''

    bad_cell_index = list() 
    pp = [shapely.geometry.Point(p) for p in points ]
    vor = Voronoi(points)
    for k in np.arange(len(vor.point_region)):
        region = vor.regions[vor.point_region[k]]
        if -1 in region:
            bad_cell_index.append(k)
            
    lines = [
        shapely.geometry.LineString(vor.vertices[line])
        for line in vor.ridge_vertices
        if -1 not in line
    ]
    polys = shapely.ops.polygonize(lines)
    areas = np.zeros(len(list(polys)))
    i = 0
    for poly in shapely.ops.polygonize(lines):
        areas[i] = poly.area
        i = i +1
    area_gmm = find_mixture(areas)
    area_pred = area_gmm.predict(areas.reshape(-1,1))
    unique, counts = np.unique(area_pred, return_counts=True)
    good_class = unique[np.where(counts == max(counts))[0]]
    cut_off = min(areas[np.where(area_pred != good_class)[0]])

    for poly in shapely.ops.polygonize(lines):
        if poly.area > cut_off:   
            bad_cell_index.append(int(np.where([poly.contains(pts) for pts in pp])[0])) 
    return bad_cell_index

def count_component(cellGraph, newLabels, locs):
    '''
    Returns number of subgraphs.
    
    :param points: cellGraph: shape (n,3); newLabels: ndarray shape (n,); locs: shape (n, 2) 

    :rtype: scalar. 
    
    '''
    G=nx.Graph()
    tempGraph = cellGraph.copy()
    tempGraph = np.apply_along_axis(remove_egdes, 1, tempGraph, newLabels)
    G.add_edges_from(tempGraph[np.where(tempGraph[:,2] == 1)[0],0:2].astype(np.int32))
    com = sorted(nx.connected_components(G), 
                                  key = len, reverse=True)   
    sum_nodes = 0
    for cc in com:
        sum_nodes = sum_nodes + len(cc)
    t_com = len(com) + locs.shape[0] - sum_nodes 
    return t_com

def compute_spatial_genomewise(locs, data_norm, size_factor=10, 
                      unary_scale_factor=100, label_cost=10, algorithm='expansion'):
    '''
    identify spatial genes on genome-scale.
    
    :param points: locs: shape (n, 2) ; data_norm: shape (n, m);  
                unary_scale_factor=100; label_cost=10; algorithm='expansion'.
    :rtype: p_values: list, genes: list, smooth_factors: list, pred_labels: list. 
    
    '''    
    num_sig = 0
    genes = list()
    p_values = list()
    smooth_factors = list()
    pred_labels = list()
    unary_scale_factor=unary_scale_factor
    size_factor=size_factor
    label_cost=label_cost
    algorithm=algorithm
#    alpha_list = compute_alpha_shape(locs)
    data_norm=data_norm
    exp =  data_norm.iloc[:,1]
    exp=(log1p(exp)).values
    cellGraph = create_graph_with_weight(locs, exp)

    for i in np.arange(data_norm.shape[1]):
        exp =  data_norm.iloc[:,i]
        if len(np.where(exp > 0)[0]) >= 10:
#            print(data_norm.columns[i])
            exp=(log1p(exp)).values
#            cellGraph = create_graph_with_weight(locs, exp)
            
            start_factor = 10
            end_factor = 50
            newLabels, gmm = cut_graph_general(cellGraph, exp, unary_scale_factor, 
                                               start_factor, label_cost, algorithm)
            t_com = count_component(cellGraph, newLabels, locs)

            if t_com <= size_factor:
                p, node, com = compute_p_CSR(newLabels, gmm, exp, cellGraph)
                if len(p) > 0:
                    p_values.append(min(p))
                    genes.append(data_norm.columns[i])
                    smooth_factors.append(start_factor)
                    pred_labels.append(newLabels)
#                    print('first tcom < size_f ', start_factor, data_norm.columns[i], t_com, size_factor)
            else:
                while (t_com > size_factor and start_factor<=end_factor):
                    start_factor = start_factor + 5
                    newLabels, gmm = cut_graph_general(cellGraph, exp, unary_scale_factor, 
                                               start_factor, label_cost, algorithm)
                    t_com = count_component(cellGraph, newLabels, locs)
                    if t_com <= size_factor:
                        p, node, com = compute_p_CSR(newLabels, gmm, exp, cellGraph)                
                        if len(p) > 0:
                            p_values.append(min(p))
                            genes.append(data_norm.columns[i])     
                            smooth_factors.append(start_factor)
                            pred_labels.append(newLabels)
                            break
                if (t_com > size_factor):
                    newLabels, gmm = cut_graph_general(cellGraph, exp, unary_scale_factor, 
                                               start_factor, label_cost, algorithm)
                    p, node, com = compute_p_CSR(newLabels, gmm, exp, cellGraph)
                    if len(p) > 0:
                        p_values.append(min(p))
                        genes.append(data_norm.columns[i])
                        smooth_factors.append(start_factor)
                        pred_labels.append(newLabels)
 
    return p_values, genes, smooth_factors, pred_labels


def estimate_smooth_factor_genomewise_simple(locs, data_norm, unary_scale_factor=100, 
                      smooth_factor=50, label_cost=10, algorithm='expansion'):
    '''
    Estimate smooth factor for the data set.
    
    :param points: locs, ndarray shape (n ,2), coords of cells;
                    data_norm, dataframe (m ,n), normalized gene expression; 
                    smooth_factor=50; label_cost=10; algorithm='expansion'.
    :rtype: genes, p_values, subnet_num. Three lists. 
    
    '''    
    num_sig = 0
    p_values = list()
    genes = list()
    subnet_num = list()
    unary_scale_factor=unary_scale_factor
    smooth_factor=smooth_factor
    label_cost=label_cost
    algorithm=algorithm
    exp =  data_norm.iloc[:,1]
    exp=(log1p(exp)).values
    cellGraph = create_graph_with_weight(locs, exp)    
#    alpha_list = compute_alpha_shape(locs)
    data_norm=data_norm
    for i in np.arange(data_norm.shape[1]):
        exp =  data_norm.iloc[:,i]
#        if i % 500 == 0:
#            print(data_norm.shape[1], i, data_norm.columns[i])
        if len(np.where(exp > 0)[0]) >= 10:
            exp=(log1p(exp)).values
#            cellGraph = create_graph_with_weight(locs, exp)
            newLabels, gmm = cut_graph_general(cellGraph, exp, unary_scale_factor, 
                                               smooth_factor, label_cost, algorithm)
            p, node, com = compute_p_CSR_simple(newLabels, gmm, exp, cellGraph)
            sum_nodes = 0
            for cc in com:
                sum_nodes = sum_nodes + len(cc)
            t_com = len(com) + locs.shape[0] - sum_nodes
            if len(p) > 0:
                p_values.append(min(p))
                subnet_num.append(t_com)
                genes.append(data_norm.columns[i]) 

    return genes, p_values, subnet_num


def subplot_voronoi_boundary_new(geneID, coord, count, 
                          classLabel, p, ax, fdr=False, show_point=False,
                          point_size = 0.1,  
                          line_colors = 'k', class_line_width = 1, 
                          line_width = 0.2, line_alpha = 1.0):
    '''
    plot spatial expression as voronoi tessellation
    highlight boundary between classes
    
    :param file: geneID; spatial coordinates shape (n, 2); count: shape (n,); 
                predicted cell class calls shape (n,); p: p value scalar; ax for plot;
                
    '''
    points = coord
    locs = coord
    count = count
    labels = classLabel
    vor = Voronoi(points)

    minima = min(count)
    maxima = max(count)
    norm = mpl.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.PiYG)
    mapper.set_array(count)

    voronoi_plot_2d(vor, ax = ax, show_points=show_point, show_vertices=False, 
                    line_colors = line_colors, line_width = line_width, 
                    line_alpha = line_alpha, point_size = point_size)
    for r in range(len(vor.point_region)):
        region = vor.regions[vor.point_region[r]]
        if not -1 in region:
            polygon = [vor.vertices[i] for i in region]
            ax.fill(*zip(*polygon), color=mapper.to_rgba(count[r]))
   
    # plot ridge between two points 
    center = points.mean(axis=0)
    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = np.asarray(simplex)
#        print(pointidx)
        # check whether the two points belong to different classes
        # and plot with color accoding to the classes
        if np.any(simplex < 0):
            if labels[pointidx[0]] != labels[pointidx[1]]:
                i = simplex[simplex >= 0][0] # finite end Voronoi vertex
                t = points[pointidx[1]] - points[pointidx[0]]  # tangent
                t = t / np.linalg.norm(t)
                n = np.array([-t[1], t[0]]) # normal
                midpoint = points[pointidx].mean(axis=0)
                far_point = vor.vertices[i] + np.sign(np.dot(midpoint - center, n)) * n * 100
                ax.plot([vor.vertices[i,0], far_point[0]],
                    [vor.vertices[i,1], far_point[1]], 'k--', lw=class_line_width)
        else:
            if labels[pointidx[0]] != labels[pointidx[1]]:
                ax.plot(vor.vertices[simplex, 0], 
                        vor.vertices[simplex, 1], 'k-', lw=class_line_width)
                
    ax.scatter(locs[:,0], locs[:,1], marker='.',
               s=point_size, c='black', zorder=10)
    ax.set_xlim(min(points[:,0])-0.5, max(points[:,0]) + 0.5); 
    ax.set_ylim(min(points[:,1])-0.5, max(points[:,1]) + 0.5)
#    ax.colorbar(mapper)
    if fdr:
        titleText = geneID + '\n' + 'FDR = ' + str("{:.2e}".format(p))
    else:
        titleText = geneID + '  ' + 'P = ' + str("{:.2e}".format(p))    
    ax.set_title(titleText, fontname="Arial", fontsize=6)
#    ax.axis('off')
#    ax.set_xlabel('X coordinate')
#    ax.set_ylabel('Y coordinate')

def subplot_voronoi_boundary(geneID, coord, count, 
                          classLabel, p, ax, fdr=False, point_size = 0.5,  
                          line_colors = 'k', class_line_width = 1.5, 
                          line_width = 0.25, line_alpha = 1.0):
    '''
    plot spatial expression as voronoi tessellation
    highlight boundary between classes
    
    :param file: geneID; spatial coordinates (n, 2); normalized gene expression: shape (n,);
            predicted cell class calls shape (n,); p_value; ax number;
    '''
    points = coord
    count = count
    labels = classLabel
    vor = Voronoi(points)

    minima = min(count)
    maxima = max(count)
    norm = mpl.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.PiYG)
    mapper.set_array(count)

    voronoi_plot_2d(vor, ax = ax, show_points=True, show_vertices=False, 
                    line_colors = line_colors, line_width = line_width, 
                    line_alpha = line_alpha, point_size = point_size)
    for r in range(len(vor.point_region)):
        region = vor.regions[vor.point_region[r]]
        if not -1 in region:
            polygon = [vor.vertices[i] for i in region]
            ax.fill(*zip(*polygon), color=mapper.to_rgba(count[r]))
   
    # plot ridge between two points 
    center = points.mean(axis=0)
    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = np.asarray(simplex)
#        print(pointidx)
        # check whether the two points belong to different classes
        # and plot with color accoding to the classes
        if np.any(simplex < 0):
            if labels[pointidx[0]] != labels[pointidx[1]]:
                i = simplex[simplex >= 0][0] # finite end Voronoi vertex
                t = points[pointidx[1]] - points[pointidx[0]]  # tangent
                t = t / np.linalg.norm(t)
                n = np.array([-t[1], t[0]]) # normal
                midpoint = points[pointidx].mean(axis=0)
                far_point = vor.vertices[i] + np.sign(np.dot(midpoint - center, n)) * n * 100
                ax.plot([vor.vertices[i,0], far_point[0]],
                    [vor.vertices[i,1], far_point[1]], 'k--', lw=class_line_width)
        else:
            if labels[pointidx[0]] != labels[pointidx[1]]:
                ax.plot(vor.vertices[simplex, 0], 
                        vor.vertices[simplex, 1], 'k-', lw=class_line_width)
                
    ax.set_xlim(min(points[:,0])-0.5, max(points[:,0]) + 0.5); 
    ax.set_ylim(min(points[:,1])-0.5, max(points[:,1]) + 0.5)
#    ax.colorbar(mapper)
    if fdr:
        titleText = geneID + '\n' + 'fdr: ' + str("{:.2e}".format(p))
    else:
        titleText = geneID + '\n' + 'p_value: ' + str("{:.2e}".format(p))    
    ax.set_title(titleText, fontname="Arial", fontsize=12)
#    ax.axis('off')
#    ax.set_xlabel('X coordinate')
#    ax.set_ylabel('Y coordinate')

def subplot_voronoi(geneID, coord, count, ax, point_size = 0.5,
                    line_colors = 'k', line_width = 0.25, line_alpha = 1.0):
    '''
    plot spatial expression as voronoi tessellation
    
    :param file: geneID; spatial coordinates shape (n, 2); normalized count: shape (n, ); 
    '''
    points = coord
    tempPoints = points
    hull = ConvexHull(points)
    polygon = shapely.geometry.Polygon(points[hull.vertices])
    for simplex in hull.simplices:
        point = rotate(points[simplex[0]], points[simplex[1]], math.radians(60))
        if polygon.contains(shapely.geometry.Point(point)):
            point = rotate(points[simplex[0]], points[simplex[1]], math.radians(-60))
        np.append(tempPoints, [point], axis = 0)
    vorTemp = Voronoi(tempPoints)
    vor = Voronoi(points) 
    minima = min(count)
    maxima = max(count)
    norm = mpl.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.PiYG)
    mapper.set_array(count)
    voronoi_plot_2d(vor, show_points=True, ax=ax, point_size=point_size,
                    show_vertices=False, line_colors = line_colors, 
                    line_width = line_width, line_alpha = line_alpha)
    for r in range(len(vor.point_region)):
        region = vor.regions[vor.point_region[r]]
        if not -1 in region:
            polygon = [vor.vertices[i] for i in region]
            ax.fill(*zip(*polygon), color=mapper.to_rgba(count[r]))

#    plt.colorbar(mapper)
    titleText = geneID + '\n' + 'No significant graph cut' 
    ax.set_title(titleText, fontname="Arial", fontsize=8)
#    ax.axis('off')
#    plt.xlabel('X coordinate')
#    plt.ylabel('Y coordinate')
#    plt.show()

def visualize_graph(geneID, locs, exp, point_size=1):
    '''
    plot graph representation of cell coordinates 
    
    :param file: geneID; spatial coordinates (n, 2); normalized count: shape (n, ); 
    point_size = 1.
    '''    
    plt.figure(figsize=(6,2.5),dpi=300)
    p1 = plt.subplot(121)
    p2 = plt.subplot(122)
    cellGraph = create_graph_with_weight(locs, exp)
    p1.scatter(locs[:,0], locs[:,1], s=1, color='black')
    for i in np.arange(cellGraph.shape[0]):
        x = (locs[int(cellGraph[i,0]), 0], locs[int(cellGraph[i,1]), 0]) 
        y = (locs[int(cellGraph[i,0]), 1], locs[int(cellGraph[i,1]), 1])     
#        p1.plot(x, y, 'o',fillstyle='full', color='black', markersize=0.1, markeredgewidth=0.0)# color='black', linewidth=0.5)
        p1.plot(x, y, color='black', linewidth=0.5)
#    for j, p in enumerate(locs):#       
#        if j == 225 or j == 155 or j == 219:
#            plt.text(p[0]-0.03, p[1]+0.03, j, ha='right') # label the points    

    subplot_voronoi(geneID, locs, exp, ax= p2, point_size = point_size)
    plt.show()

def visualize_graph_boundary(geneID, locs, exp, classLabel, 
                             point_size=1, class_line_width = 2, fileName=None):
    '''
    plot graph representation of cell coordinates, highlight boundaries of graph cut
    
    :param file: geneID; spatial coordinates (n, 2); normalized count: shape (n, ); 
    point_size = 1;class_line_width = 2; fileName=None
    '''    
    labels = classLabel
    plt.figure(figsize=(6,2.5),dpi=300)
    p1 = plt.subplot(121)
    p2 = plt.subplot(122)
    cellGraph = create_graph_with_weight(locs, exp)
    p1.scatter(locs[:,0], locs[:,1], s=1, color='black')
    for i in np.arange(cellGraph.shape[0]):
        x = (locs[int(cellGraph[i,0]), 0], locs[int(cellGraph[i,1]), 0]) 
        y = (locs[int(cellGraph[i,0]), 1], locs[int(cellGraph[i,1]), 1])     
#        p1.plot(x, y, 'o',fillstyle='full', color='black', markersize=0.1, markeredgewidth=0.0)# color='black', linewidth=0.5)
        p1.plot(x, y, color='black', linewidth=0.5)
    points = locs
    vor = Voronoi(points)
    # plot ridge between two points 
    center = points.mean(axis=0)
    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = np.asarray(simplex)
#        print(pointidx)
        # check whether the two points belong to different classes
        # and plot with color accoding to the classes
        if np.any(simplex < 0):
            if labels[pointidx[0]] != labels[pointidx[1]]:
                i = simplex[simplex >= 0][0] # finite end Voronoi vertex
                t = points[pointidx[1]] - points[pointidx[0]]  # tangent
                t = t / np.linalg.norm(t)
                n = np.array([-t[1], t[0]]) # normal
                midpoint = points[pointidx].mean(axis=0)
                far_point = vor.vertices[i] + np.sign(np.dot(midpoint - center, n)) * n * 100
                p1.plot([vor.vertices[i,0], far_point[0]],
                    [vor.vertices[i,1], far_point[1]], 'k--', lw=class_line_width)
        else:
            if labels[pointidx[0]] != labels[pointidx[1]]:
                p1.plot(vor.vertices[simplex, 0], 
                        vor.vertices[simplex, 1], 'k-', lw=class_line_width)
    p1.set_xlim(min(points[:,0])-0.5, max(points[:,0]) + 0.5); 
    p1.set_ylim(min(points[:,1])-0.5, max(points[:,1]) + 0.5)
#    for j, p in enumerate(locs):#       
#        if j == 225 or j == 155 or j == 219:
#            plt.text(p[0]-0.03, p[1]+0.03, j, ha='right') # label the points    
    subplot_voronoi(geneID, locs, exp, ax= p2, point_size = point_size)
    if fileName != None:
        plt.savefig(fileName)
    plt.show()
    
def visualize_spatial_genes(df, locs, data_norm, point_size= 0.5):
    '''
    plot Voronoi tessellation of cells, highlight boundaries of graph cut
    
    :param file: df: dataframe of graph cut results; locs: spatial coordinates (n, 2); data_norm: normalized count: shape (n, m); 
    point_size = 0.5; 
    '''    
#    for i in (np.arange(df.shape[0])):

    i = 0
    while i < df.shape[0]:
        plt.figure(figsize=(6,2.5), dpi=300)
        p1 = plt.subplot(121)
        p2 = plt.subplot(122)

        geneID = df.index[i]
        exp =  data_norm.loc[:,geneID]
        exp=(log1p(exp)).values
        best_Labels = np.array((df.loc[geneID,]).astype(np.int32)[3:].values)
        subplot_voronoi_boundary(geneID, locs, exp, best_Labels,
                                 df.loc[geneID,].fdr, ax=p1, 
                                 fdr=True, point_size = point_size, class_line_width=2)
        i = i + 1
        if i < df.shape[0]:
            geneID = df.index[i]
            exp =  data_norm.loc[:,geneID]
            exp=(log1p(exp)).values
            best_Labels = np.array((df.loc[geneID,]).astype(np.int32)[3:].values)
            subplot_voronoi_boundary(geneID, locs, exp, best_Labels,
                                 df.loc[geneID,].fdr, ax=p2, fdr=True, 
                                     point_size = point_size)    
        else:
            p2.axis('off')
        plt.show()
        i= i + 1
    

def subplot_voronoi_boundary_12x18(geneID, coord, count, 
                          classLabel, p, ax, fdr=False, point_size = 0.5,  
                          line_colors = 'k', class_line_width = 0.8, 
                          line_width = 0.05, line_alpha = 1.0):
    '''
    plot spatial expression as voronoi tessellation
    highlight boundary between classes
    
    :param file: geneID; coord: spatial coordinates (n, 2); count: normalized gene expression shape (n, );
        predicted cell class calls (n); p: graph cut p-value. 
    '''
    points = coord
    count = count
    labels = classLabel
    vor = Voronoi(points)

    minima = min(count)
    maxima = max(count)
    norm = mpl.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.PiYG)
    mapper.set_array(count)

    voronoi_plot_2d(vor, ax = ax, show_points=True, show_vertices=False, 
                    line_colors = line_colors, line_width = line_width, 
                    line_alpha = line_alpha, point_size = point_size)
    for r in range(len(vor.point_region)):
        region = vor.regions[vor.point_region[r]]
        if not -1 in region:
            polygon = [vor.vertices[i] for i in region]
            ax.fill(*zip(*polygon), color=mapper.to_rgba(count[r]))
   
    # plot ridge between two points 
    center = points.mean(axis=0)
    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = np.asarray(simplex)
#        print(pointidx)
        # check whether the two points belong to different classes
        # and plot with color accoding to the classes
        if np.any(simplex < 0):
            if labels[pointidx[0]] != labels[pointidx[1]]:
                i = simplex[simplex >= 0][0] # finite end Voronoi vertex
                t = points[pointidx[1]] - points[pointidx[0]]  # tangent
                t = t / np.linalg.norm(t)
                n = np.array([-t[1], t[0]]) # normal
                midpoint = points[pointidx].mean(axis=0)
                far_point = vor.vertices[i] + np.sign(np.dot(midpoint - center, n)) * n * 100
                ax.plot([vor.vertices[i,0], far_point[0]],
                    [vor.vertices[i,1], far_point[1]], 'k--', lw=class_line_width)
        else:
            if labels[pointidx[0]] != labels[pointidx[1]]:
                ax.plot(vor.vertices[simplex, 0], 
                        vor.vertices[simplex, 1], 'k-', lw=class_line_width)
                
    ax.set_xlim(min(points[:,0])-0.5, max(points[:,0]) + 0.5); 
    ax.set_ylim(min(points[:,1])-0.5, max(points[:,1]) + 0.5)
#    ax.colorbar(mapper)
    if fdr:
        titleText = geneID + ' ' + '' + str("{:.1e}".format(p))
    else:
        titleText = geneID + ' ' + 'p_value: ' + str("{:1e}".format(p))    
    ax.set_title(titleText, fontname="Arial", fontsize=3.5, y = 0.85)
    
    
def multipage_pdf_visualize_spatial_genes_12x18(df, locs, data_norm, fileName, 
                     point_size=0.5):
    '''
    save spatial expression as voronoi tessellation to pdf highlight boundary between classes
    format: 12 by 18.
    :param file: df: dataframe for graph cuts results shape (k, m); locs: spatial coordinates (n, 2); data_norm: normalized gene expression shape(n, m);
        pdf filename; point_size=0.5. 
    '''    
    points = locs
    geneID = df.index[0]
    exp =  data_norm.loc[:,geneID]
    exp=(log1p(exp)).values
    cellGraph = create_graph_with_weight(locs, exp)
    count = exp
    vor = Voronoi(points)

    nb_plots = int(df.shape[0])
    numCols = 12  
    numRows = 18
    nb_plots_per_page =  numCols*numRows
    t_numRows = int(df.shape[0]/numCols) + 1
    
    with PdfPages(fileName) as pdf:    
#    fig, axs = plt.subplots(numRows, numCols, figsize = (15, fsize), constrained_layout=True)
        for i in np.arange(df.shape[0]):
            if i % nb_plots_per_page == 0:
                fig, axs = plt.subplots(numRows, numCols, # 8 11
                                    figsize = (8, 11))   
                fig.subplots_adjust(hspace=0.3, wspace=0.3,
                                top=0.925, right=0.925, bottom=0.075, left = 0.075)
                  
            geneID = df.index[i]
            exp =  data_norm.loc[:,geneID]
            exp=(log1p(exp)).values
            if np.isnan(df.loc[geneID,].fdr):
                best_Labels = np.zeros(data_norm.shape[0])
            else:
                best_Labels = np.array((df.loc[geneID,]).astype(np.int32)[3:].values)
            m = int(i/numCols) % numRows
            n = i % numCols 
            ax = axs[m,n]
            subplot_voronoi_boundary_12x18(geneID, locs, exp, best_Labels,
                                 df.loc[geneID,].fdr, ax=ax, fdr=True,
                                 point_size = point_size)

            if (i + 1) % nb_plots_per_page == 0 or (i + 1) == nb_plots:
                for ii in np.arange(numRows):
                    for jj in np.arange(numCols):        
                        axs[ii,jj].axis('off')

                pdf.savefig(fig)
                fig.clear()
                plt.close() 
    
def multipage_pdf_visualize_spatial_genes(df, locs, data_norm, fileName, 
                    point_size = 0.1):
    '''
    save spatial expression as voronoi tessellation to pdf highlight boundary between classes
    format: 6 by 9
    :param file: df: graph cuts results shape (k, m); locs: spatial coordinates (n, 2); data_norm: normalized gene expression shape (n, );
        pdf filename; point_size=0.5. 
    '''    
    points = locs
    geneID = df.index[0]
    exp =  data_norm.loc[:,geneID]
    exp=(log1p(exp)).values
    cellGraph = create_graph_with_weight(locs, exp)
    count = exp
    vor = Voronoi(points)
    nb_plots_per_page = 54
    nb_plots = int(df.shape[0])
    numCols = 6  # change back to 4 6
    numRows = 9
    t_numRows = int(df.shape[0]/numCols) + 1

    with PdfPages(fileName) as pdf:    
#    fig, axs = plt.subplots(numRows, numCols, figsize = (15, fsize), constrained_layout=True)
        for i in np.arange(df.shape[0]):
            if i % 54 == 0:
                fig, axs = plt.subplots(numRows, numCols, # 8 11
                                    figsize = (8, 11))   
                fig.subplots_adjust(hspace=0.5, wspace=0.2,
                                top=0.925, right=0.875, bottom=0.075)
                  
            geneID = df.index[i]
            exp =  data_norm.loc[:,geneID]
            exp=(log1p(exp)).values
            if np.isnan(df.loc[geneID,].fdr):
                best_Labels = np.zeros(data_norm.shape[0])
            else:
                best_Labels = np.array((df.loc[geneID,]).astype(np.int32)[3:].values)
            m = int(i/numCols) % 9
            n = i % numCols 
            ax = axs[m,n]
            subplot_voronoi_boundary_new(geneID, locs, exp, best_Labels,
                                 df.loc[geneID,].fdr, ax=ax, fdr=True,
                                show_point=False, point_size = point_size)

            if (i + 1) % nb_plots_per_page == 0 or (i + 1) == nb_plots:
                for ii in np.arange(numRows):
                    for jj in np.arange(numCols):        
                        axs[ii,jj].axis('off')

                pdf.savefig(fig)
                fig.clear()
                plt.close()                


            
def identify_spatial_genes(locs, data_norm, smooth_factor=10,
                           unary_scale_factor=100, label_cost=10, algorithm='expansion'):
#    pool = mp.Pool()
    '''
    main function to identify spatially variable genes
    :param file:locs: spatial coordinates (n, 2); data_norm: normalized gene expression shape(n, m);
        smooth_factor=10; unary_scale_factor=100; label_cost=10; algorithm='expansion' 
    :rtype: prediction: a dataframe
    '''    
    num_cores = mp.cpu_count()
    step=round(data_norm.shape[1]/num_cores)+1
    ttt = [data_norm.iloc[:, i*step:min((i+1)*step, data_norm.shape[1])] 
           for i in np.arange(num_cores)]
    tuples = [(l, d, u, s, c, a) for l, d, u, s, c, a in zip(repeat(locs, num_cores), ttt,
                                    repeat(smooth_factor, num_cores),
                                    repeat(unary_scale_factor, num_cores), 
                                    repeat(label_cost, num_cores),
                                    repeat(algorithm, num_cores))] 
                                    
    results = parmap.starmap(compute_spatial_genomewise, tuples,
                             pm_processes=num_cores, pm_pbar=True)
#    pool.close()
    ppp = [results[i][0] for i in np.arange(len(results))]
    p_values=reduce(operator.add, ppp)
    ggg = [results[i][1] for i in np.arange(len(results))]
    genes = reduce(operator.add, ggg)
    fff = [results[i][2] for i in np.arange(len(results))]
    s_factors = reduce(operator.add, fff)
    lll = [results[i][3] for i in np.arange(len(results))]
    pred_labels = reduce(operator.add, lll)
    fdr = multi.multipletests(np.array(p_values), method='fdr_bh')[1]
    
    labels_array = np.array(pred_labels).reshape(len(genes), pred_labels[0].shape[0])
    data_array = np.array((genes, p_values, fdr, s_factors)).T
    t_array = np.hstack((data_array, labels_array))
    c_labels = ['p_value', 'fdr', 'smooth_factor']
    for i in np.arange(labels_array.shape[1]) + 1:
        temp_label = 'label_cell_' + str(i)
        c_labels.append(temp_label)
    df = pd.DataFrame(t_array[:,1:], index=t_array[:,0], 
                      columns=c_labels)
    df = df.astype('float')
    return df

def estimate_smooth_factor(locs, data_norm, unary_scale_factor=100, 
                      label_cost=10, algorithm='expansion'):
    '''
    main function to estimate smooth factor for graph cut
    :param file:locs: spatial coordinates (n, 2); data_norm: normalized gene expression shape (n, m);
        unary_scale_factor=100; label_cost=10; algorithm='expansion' 
    :rtype: factor_df: a dataframe; optim_factor: a scalar, best smooth factor.
    '''       
    smooth_factor = 25
#    pool = mp.Pool()
    num_cores = mp.cpu_count()
    step=round(data_norm.shape[1]/num_cores)+1
    ttt = [data_norm.iloc[:, i*step:min((i+1)*step, data_norm.shape[1])] 
           for i in np.arange(num_cores)]
    tuples = [(l, d, u, s, c, a) for l, d, u, s, c, a in zip(repeat(locs, num_cores), ttt,
                                    repeat(unary_scale_factor, num_cores), 
                                    repeat(smooth_factor, num_cores),
                                    repeat(label_cost, num_cores),
                                    repeat(algorithm, num_cores))] 
                                    
    results = parmap.starmap(estimate_smooth_factor_genomewise_simple, 
                             tuples, pm_processes=num_cores, pm_pbar=True)
#    pool.close()
    ggg = [results[i][0] for i in np.arange(len(results))]
    genes=reduce(operator.add, ggg)    
    ppp = [results[i][1] for i in np.arange(len(results))]
    p_values=reduce(operator.add, ppp)
    subnet_num = [results[i][2] for i in np.arange(len(results))]
    nums = reduce(operator.add, subnet_num)
    fdr = multi.multipletests(np.array(p_values), method='fdr_bh')[1]
    data_array = np.array((genes, p_values, fdr, nums)).T
    factor_df = pd.DataFrame(data_array[:,1:], index=data_array[:,0], columns=('p_value', 'fdr', 'subnet_num'))
#    df_sel = df[df.fdr < 0.0001]
    factor_df = factor_df.astype('float')
    factor_sel = factor_df[factor_df.fdr < 0.01]
    if factor_sel.shape[0] > 10:
        optim_factor = int(np.percentile(factor_sel.subnet_num,90))
    else:
        optim_factor = int(np.mean(factor_df.subnet_num))
    return factor_df, optim_factor


    
def spatial_pca_tsne(data_norm, gene_lists, perplexity = 30):
    '''
    perform standard PCA + tsne
    :param file: data_norm: normalized gene expression; gene_lists: list shape(k,)
        perplexity = 30 
    :rtype: tsne_proj: shape (m, 2)
    '''           
    data_s = StandardScaler().fit_transform(data_norm.loc[:, gene_lists])
    pca = decomposition.PCA()
    pca.fit(data_s.T)
    pca_proj = pca.fit_transform(data_s.T)
    num_comp = np.where(np.cumsum(pca.explained_variance_)/np.sum(pca.explained_variance_) 
                    > 0.9)[0][0]

#    RS=20180824
    tsne=manifold.TSNE(n_components=2, perplexity=perplexity)
    tsne_proj = tsne.fit_transform(pca_proj[:,0:num_comp])
    return tsne_proj

def visualize_tsne_density(tsne_proj, threshold=0.001, bins=100, fileName=None):
    '''
    perform kde density estimationg for tsne projection 
    :param file: tsne_proj: shape (m, 2)
    threshold=0.001, bins=100, fileName=None
    '''   
    fig, ax = plt.subplots()
    kde = gaussian_kde(tsne_proj.T, bw_method = 'scott')
    z = kde(tsne_proj.T)    
    x = np.ma.masked_where(z > threshold, tsne_proj[:,0])
    y = np.ma.masked_where(z > threshold, tsne_proj[:,1])

    # plot unmasked points
    ax.scatter(tsne_proj[:,0], tsne_proj[:,1], c='black', marker='o', s=5)

    # get bounds from axes
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # prepare grid for density map
    xedges = np.linspace(xmin, xmax, bins)
    yedges = np.linspace(ymin, ymax, bins)
    xx, yy = np.meshgrid(xedges, yedges)
    gridpoints = np.array([xx.ravel(), yy.ravel()])

    # compute density map
    zz = np.reshape(kde(gridpoints), xx.shape)

    # plot density map
    im = ax.imshow(zz, cmap='Spectral_r', interpolation='nearest',
               origin='lower', extent=[xmin, xmax, ymin, ymax],
                  aspect='auto')
    # plot threshold contour
    cs = ax.contour(xx, yy, zz, levels=[threshold], colors='black')
    # show
    fig.colorbar(im)   

    if fileName != None:
        plt.savefig(fileName)
    plt.show()
    return z
     
     
