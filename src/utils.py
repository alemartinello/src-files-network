"""
"""
from pathlib import Path
from math import log
import networkx as nx


class FileNetwork():
    def __init__(self, sourcepath, cycles_color='red'):
        """
        """
        self.sourcepath = Path(sourcepath)
        self.filelist = [
            path.relative_to(self.sourcepath) for path in self.sourcepath.glob('**/*.py')
        ]
        self.internal = [self.name(path) for path in self.filelist]
        self.networkdict = {}
        self.cycles_color = cycles_color
        self.gen_network()

    def network_edges(self):
        edgelist = []
        for _file in self.filelist:
            edgelist += self._imports_into_edges(_file)
        return edgelist

    def network_nodes(self):
        nodes = []
        for node in self.filelist:
            node_attributes = {'type': node.suffix}
            if node_attributes['type'] == '.py':
                node_attributes['size'] = 2 + log(self.get_filesize(self.sourcepath / node)) * 2
                node_attributes['color'] = {'border': 'blue', 'background': 'rgba(122,122,122,1)'}
            nodes.append((self.name(node), node_attributes))
        return nodes

    def gen_network(self):
        di = nx.DiGraph()
        di.add_edges_from(self.network_edges())
        di.add_nodes_from(self.network_nodes())
        self.network = di
        self.highlight_cycles()
        return self

    def highlight_cycles(self):
        for cycle in nx.simple_cycles(self.network):
            i = 0
            while i < len(cycle) - 1:
                self.network[cycle[i]][cycle[i + 1]]['color'] = self.cycles_color
                i += 1
            self.network[cycle[-1]][cycle[0]]['color'] = self.cycles_color

    def _imports_into_edges(self, filepath):
        """
        """
        edgelist = []
        for imp in self.get_imports(self.sourcepath / filepath):
            internal_file = False
            for _intern in self.internal:
                if imp in _intern:
                    imp = _intern
                    internal_file = True
                    break
            if internal_file is False:
                imp = imp.split('.')[0]
                edge_dictionary = {}
            else:
                edge_dictionary = {'width': 2}
            edgelist.append((imp, self.name(filepath), edge_dictionary))
        return edgelist

    @staticmethod
    def get_imports(inputfile):
        """
        """
        with inputfile.open() as f:
            lines = f.read().splitlines()

        lines = [line for line in lines if len(line.strip()) > 0]
        importlist = [
            iline.split()[1] for iline in [
                line for line in lines if (line.strip().split()[0] == 'import') or (line.strip().split()[0] == 'from')
            ]
        ]
        return importlist

    @staticmethod
    def get_filesize(inputfile):
        with open(inputfile, 'rb') as f:
            lines = 0
            buf_size = 1024 * 1024
            read_f = f.raw.read

            buf = read_f(buf_size)
            while buf:
                lines += buf.count(b'\n')
                buf = read_f(buf_size)

        return lines

    @staticmethod
    def name(path):
        return str(path).replace('/', '.').replace('.py', '')
