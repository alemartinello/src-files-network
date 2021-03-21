import os
from itertools import cycle, islice
import gensim
import numpy as np
import pandas as pd
import codecs
import json
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import scipy.cluster.hierarchy as hierarchy
import src.fui.hierarchymod as hierarchymod
from src.fui.utils import params
from scipy.spatial.distance import pdist

class ClusterTree():
    """Build clusters from topic models using scipy.cluster.hierarchy.
    """
    
    def __init__(self, num_topics, metric='jensenshannon', method='ward', 
                 unique_scale=True, topn=None):
        """
        Saves linkage matrix `Z´ and `nodelist´
        args:
            num_topics (int): Selects LDA model.
            metric (str): Metric passed to scipy.spatial.distance.pdist 
            method (str): Method passed to scipy.cluster.hierarchy
            unique_scale (bool): Scale word proba by uniqueness
            topn (int, optional): only consider X words (don't use)
        """
        
        self.num_topics = num_topics
        self.metric = metric
        self.method = method
        self.scale = 200
        
        folder_path = os.path.join(params().paths['lda'], 
                                   'lda_model_' + str(self.num_topics))
        file_path = os.path.join(folder_path, 'trained_lda')
        self.lda_model = gensim.models.LdaMulticore.load(file_path)
        topics = self.lda_model.get_topics()
        if unique_scale:
            topics = topics/(topics.sum(axis=0))
        if topn:
            topics.sort(axis=1)
            topics = np.flip(topics,axis=1)
            topics = topics[:,0:topn]
        y = pdist(topics, metric=self.metric)
        self.Z = hierarchy.linkage(y, method=self.method)
        rootnode, self.nodelist = hierarchy.to_tree(self.Z,rd=True)
    
    def parse_topic_labels(self,name):
        """
        reads hand labeled topics from json file.
        
        """
        label_path = os.path.join(params().paths['topic_labels'], 
                                  name+str(self.num_topics)+'.json')
          
        with codecs.open(label_path, 'r', encoding='utf-8-sig') as f:
            self.labels = json.load(f)
        return self.labels
    
    
    def _get_children(self, id):
        """
        Recursively get all children of parent node `id´
        """
        if not self.nodelist[id].is_leaf():
            for child in [self.nodelist[id].get_left(), self.nodelist[id].get_right()]:
                yield child
                for grandchild in self._get_children(child.id):
                    yield grandchild
                    
    def children(self):
        """Returns a dict with k, v: parent: [children]. Does not include leaf nodes.
        """
        self.children = {}
        for i in range(self.num_topics,len(self.nodelist)):
            self.children[i] = [child.id for child in self._get_children(i)]
        return self.children
    
    def _get_topic_sums(self):
        """Get sum of topic probabilities across articles.
        """        
        df = pd.read_hdf(params().paths['doc_topics']+'doc_topics_u_count_extend.h5', 'table')
        df = df.iloc[:,0:self.num_topics].values.tolist()
        
        self.topic_sums = np.array(df).sum(axis=0)
    
    def get_node_weights(self):
        self._get_topic_sums()
        self.node_weights = [None]*len(self.nodelist)
        for node in self.nodelist:
            if node.is_leaf():
                self.node_weights[node.id] = self.topic_sums[node.id]
            else:
                self.node_weights[node.id] = 0
                children = [c.id for c in self._get_children(node.id) if c.is_leaf()]
                for c in children:
                    self.node_weights[node.id] += self.topic_sums[c]   
        self.node_weights = pd.Series(self.node_weights)
        self.node_weights = self.node_weights/np.max(self.node_weights)
                    
    def _colorpicker(self,k):
        """Returns an NB color to visually group similar topics in dendrogram
        """
        NB_colors = [(0, 123, 209),
                     (146, 34, 156),
                     (196, 61, 33),
                     (223, 147, 55),
                     (176, 210, 71)] 
        
        # Get flat clusters for grouping
        self.flat_clusters(n=self.colors)
        clist = list(islice(cycle(NB_colors), len(self.L)))
        for c,i in enumerate(list(self.L)):
            if k in [child.id for child in self._get_children(i)]:
                color = clist[c]
                return f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
    
        # Gray is default color
        return "#666666"
    
    def _labelpicker(self,k):
        #_list = self.labels[str(k)]
        #_list.append(str(k))
        #return ', '.join(_list)
        return self.labels[str(k)]
        
    def dendrogram(self,w=12,h=17,colors=10,
                   color_labels=True,weight_nodes=True,annotate=True):
        """
        Draws dendrogram
        :colors: Approx. no of color clusters in figure.
        """
        self.labels = self.parse_topic_labels('labels')
        self.colors = colors
        fig = plt.figure(figsize=(w,h)) 
        #plt.title("Topic Dendrogram")
        plt.xlabel("Distance")
        #plt.ylabel("Topic")
        
        R = hierarchymod.dendrogram(self.Z,
                       orientation='right',
                       #labels=labelList,
                       distance_sort='descending',
                       show_leaf_counts=False,
                       no_plot=False,
                       leaf_label_func=self._labelpicker,
                       #color_threshold=2.0*np.max(self.Z[:,2])
                       link_color_func=self._colorpicker)
        
        self.ax = plt.gca()
        
        if weight_nodes:
            self.get_node_weights()
            
            #assumes orientation left or right
            self.lines = []
            for (xline, yline) in zip(R['dcoord'], R['icoord']):
                coords = list(zip(xline, yline))
                self.lines.append(coords)
            for i,line in enumerate(self.lines):
                coord_array = np.array(line,dtype=float)
                line.append(coord_array)
                line.append(R['i_list'][i])
                
            i_dict = {}
            new_colls = []
            num_colls = len(self.ax.collections)
            for i,c in enumerate(self.ax.collections):
                i_dict[i] = []
                segments = []
                widths = []
                color = c.get_color()
                for j,p in enumerate(c.get_paths()):
                    for line in self.lines:
                            if np.equal(line[4], p.vertices).all():
                                i_dict[i].append(line[5])
                                s, w = self.segment_path(p.vertices,line[5])
                    segments.extend(s)
                    widths.extend(w)
                coll = LineCollection(segments)
                coll.set_color(color)
                coll.set_linewidths(widths)
                new_colls.append(coll)
                
            # replace old line collections
            for c in new_colls:
                self.ax.add_collection(c)
            self.ax.collections = self.ax.collections[num_colls:]
        
        if color_labels:
            self.cluster_idxs = {}
            for c, pi in zip(R['color_list'], R['icoord']):
                for leg in pi[1:3]:
                    i = (leg - 5.0) / 10.0
                    if abs(i - int(i)) < 1e-5:
                        self.cluster_idxs[int(i)] = c
            
            ylbls = self.ax.get_ymajorticklabels()
            for c,y in enumerate(ylbls):
                y.set_color(self.cluster_idxs[c])
            
            #tempfix
            self.ax.get_ymajorticklabels()[11].set_color(self.cluster_idxs[12])
            
        self.ax.set_xlim(left=0.6)
        if annotate:
            #self.ax.annotate("Fiscal policy", (1.08, 20))
            self.ax.annotate("Fiscal policy\nand corporate finance", (1.12, 43))
            self.ax.annotate("Financial markets", (1.13, 133))
            #self.ax.annotate("Politics, domestic", (0.94,179))
            self.ax.annotate("Entertainment", (1.14, 225))
            self.ax.annotate("Labor market, career \nand organization", (1.03, 295))
            self.ax.annotate("Politics", (1.085, 383))
            #self.ax.annotate("Crime", (0.925,405))
            self.ax.annotate("Sports", (1.05, 460))
            self.ax.annotate("US & UK", (1.0, 600))
            self.ax.annotate("Industry and trade", (1.08, 770))
            #self.ax.annotate("Environment", (1.01, 777))
                
        plt.tight_layout()
        fig.savefig(os.path.join(params().paths['lda'], 
                                   'dendrogram'+str(self.num_topics)+'.pdf'), dpi=300)
        fig.savefig(os.path.join(params().paths['lda'],
                                   'dendrogram'+str(self.num_topics)+'.png'), dpi=300)

        plt.show()
        return fig, self.ax, R
    
        
    def segment_path(self,verts,nodes):
        """
        Converts original U-path into three segments, bottom, top and bar.
        Sets linewidth of top and bottom each segment proportional to node weight.
        args:
            verts: np.array with vertices for path
            nodes: list of ids of the node represented by top and bottom
        returns:
            (list,list) of segments vertices and linewidths
        """
        bottom = np.array(verts[0:2], copy=True)
        top = np.array(verts[2:4], copy=True)
        bar = np.array(verts[1:3], copy=True)
        
        w_b = self.node_weights[nodes[0]]*self.scale
        w_a = self.node_weights[nodes[1]]*self.scale
        w_mid = 1.5
         
        # make pretty by extending top and bottom over the bar        
        top[0][0] += 0.0007
        bottom[1][0] += 0.0007

        segment = [bottom,top,bar]
        widths = [w_b,w_a,w_mid]
        
        return segment, widths
    
    def flat_clusters(self,n=8,init=1,criterion='maxclust'):
        """
        Returns flat clusters from the linkage matrix :Z:
        """
        if criterion is 'distance':
            self.T = hierarchy.fcluster(self.Z,init,criterion='distance')
            a = 0
            while a < 20:
                if self.T.max() < n:
                    init = init-0.02
                    a += 1
                elif self.T.max() > n:
                    init = init+0.02
                    a += 1
                else:
                    self.L, self.M = hierarchy.leaders(self.Z,self.T)
                    return self.T
                self.T = hierarchy.fcluster(self.Z,init,criterion='distance')
            self.L, self.M = hierarchy.leaders(self.Z,self.T)
            return self.T
        elif criterion is 'inconsistent':
            self.T = hierarchy.fcluster(self.Z,criterion='inconsistent')
            self.L, self.M = hierarchy.leaders(self.Z,self.T)
            return self.T
        elif criterion is 'maxclust':
            self.T = hierarchy.fcluster(self.Z,t=n,criterion='maxclust')
            self.L, self.M = hierarchy.leaders(self.Z,self.T)
            return self.T
        else:
            print('Criteria not implemented')
            return 0
        
        
if __name__ == '__main__':
    cl90 = ClusterTree(90,metric='cosine')
    fig, ax, R = cl90.dendrogram(colors=8)