"""
"""
from pathlib import Path
from math import log
from pyvis.network import Network
import networkx as nx
import click


class FileNetwork():
    def __init__(self, sourcepath, cycles_color='red', usegitignore=True):
        """
        """
        self.sourcepath = Path(sourcepath)
        self.usegitignore = usegitignore
        self.filelist = self._get_pyfilelist(self.sourcepath, self.usegitignore)
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
                node_attributes['color'] = {'border': "rgba(0,70,10,1)", "background": "rgba(0, 120, 20 ,1)"}
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
                edge_dictionary = {'width': 4}
            edgelist.append((imp, self.name(filepath), edge_dictionary))
        return edgelist

    @staticmethod
    def _get_pyfilelist(srcpath, usegitignore=True):
        """
        """
        gitignorefile = srcpath / Path('.gitignore')
        if usegitignore and gitignorefile.exists():
            with gitignorefile.open() as f:
                lines = f.read().splitlines()
            gitignore = [
                Path(line) for line in lines if not line.strip().startswith('#') and len(line.strip()) > 1 and Path(line).suffix == ''
            ] + [Path('.git')]
            viablepaths = [p for p in srcpath.glob('*/') if p.is_dir() and p not in gitignore]
            filelist = set().union(*[set(p.glob('**/*.py')) for p in viablepaths])
        else:
            filelist = srcpath.glob('**/*.py')

        return [p.relative_to(srcpath) for p in filelist]

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


def get_pyvis_options():
    with open('src/pyvis_options.json', 'r') as f:
        options = f.read()
    return f"""var options = {str(options)}"""


def plot_network(path, saveas=None):
    if saveas is None:
        saveas = '_srcnetwork.html'
    fn = FileNetwork(path)
    nt = Network('1500px', '1500px')
    nt.toggle_physics(True)
    nt.from_nx(fn.network)
    nt.set_options(get_pyvis_options())
    nt.show(f'{saveas}.html')
    return


@click.command()
@click.argument('path', type=click.Path(exists=True))
@click.option('--saveas', default='_scrnetwork.html', show_default=True, type=click.File(mode='w'))
def plotnetwork(path, saveas):
    click.echo(path)
    click.echo(saveas)


if __name__ == '__main__':
    plotnetwork()
