"""
"""
from pathlib import Path
from math import log
from pyvis.network import Network
import networkx as nx
import click


class FileNetwork:
    """
    Maps `.py` files in a folder into a network based on cross-imports.
    Automatically highlights circular imports, and ignores folders specified in
    `.gitignore`

    Parameters:
    ---
    sourcepath: Path. Folder to map

    cycles_color: str. Color with which to highlight edges in the network forming
    circular imports. Can be a rgb string such as `rgba(0, 120, 20 ,1)`. Default: `'red'`

    usegitignore: Bool. Whether to ignore folders specified in gitignore. Default: `True`
    """

    def __init__(self, sourcepath, cycles_color="red", usegitignore=True):
        """
        Initializes network
        """
        self.sourcepath = Path(sourcepath)
        self.usegitignore = usegitignore
        self.filelist = self._get_pyfilelist(self.sourcepath, self.usegitignore)
        self.internal = [self.name(path) for path in self.filelist]
        self.networkdict = {}
        self.cycles_color = cycles_color
        self.gen_network()

    def network_edges(self):
        """
        Extracts network edges
        """
        edgelist = []
        for _file in self.filelist:
            edgelist += self._imports_into_edges(_file)
        return edgelist

    def network_nodes(self):
        """
        Extracts network nodes and their attributes
        """
        nodes = []
        for node in self.filelist:
            node_attributes = {"type": node.suffix}
            if node_attributes["type"] == ".py":
                node_attributes["size"] = (
                    log(self.get_filesize(self.sourcepath / node) + 25) * 2
                )
                node_attributes["color"] = {
                    "border": "rgba(0,70,10,1)",
                    "background": "rgba(0, 120, 20 ,1)",
                }
            nodes.append((self.name(node), node_attributes))
        return nodes

    def gen_network(self):
        """
        Construct the Directed Graph representing cross-imports
        """
        di = nx.DiGraph()
        di.add_edges_from(self.network_edges())
        di.add_nodes_from(self.network_nodes())
        self.network = di
        self.highlight_cycles()
        return self

    def highlight_cycles(self):
        """
        Finds circular imports in the network
        """
        for cycle in nx.simple_cycles(self.network):
            i = 0
            while i < len(cycle) - 1:
                self.network[cycle[i]][cycle[i + 1]]["color"] = self.cycles_color
                i += 1
            self.network[cycle[-1]][cycle[0]]["color"] = self.cycles_color

    def _imports_into_edges(self, filepath):
        """
        Loops through python files in the folder and transforms import
        statements into edges
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
                imp = imp.split(".")[0]
                edge_dictionary = {}
            else:
                edge_dictionary = {"width": 4}
            edgelist.append((imp, self.name(filepath), edge_dictionary))
        return edgelist

    @staticmethod
    def _get_pyfilelist(srcpath, usegitignore=True) -> list:
        """
        Gets files in `srcpath` to represent in a network
        """
        gitignorefile = srcpath / Path(".gitignore")
        if usegitignore and gitignorefile.exists():
            with gitignorefile.open('r') as f:
                lines = f.read().splitlines()
            gitignore = [
                srcpath / Path(line)
                for line in lines
                if not line.strip().startswith("#")
                and len(line.strip()) > 1
                and Path(line).suffix == ""
            ] + [srcpath / Path(".git")]
            viablepaths = [
                p for p in srcpath.glob("*/") if p.is_dir() and p not in gitignore
            ]
            filelist = set().union(*[set(p.glob("**/*.py")) for p in viablepaths])
            filelist = filelist.union(*[set(srcpath.glob('*.py'))])
        else:
            filelist = srcpath.glob("**/*.py")
        return [p.relative_to(srcpath) for p in filelist]

    @staticmethod
    def get_imports(inputfile) -> list:
        """
        Finds import statements in a `.py` file
        """
        with inputfile.open('rb') as f:
            lines = f.read().decode(errors='replace').splitlines()

        lines = [line for line in lines if len(line.strip()) > 0]
        importlist = [
            iline.split()[1]
            for iline in [
                line.strip()
                for line in lines
                if (line.strip().split()[0] == "import")
                or (line.strip().split()[0] == "from")
            ]
            if len(iline.split()) > 1
        ]
        return importlist

    @staticmethod
    def get_filesize(inputfile) -> int:
        """
        Returns the legth of a file in lines
        """
        with open(inputfile, "rb") as f:
            lines = 0
            buf_size = 1024 * 1024
            read_f = f.raw.read

            buf = read_f(buf_size)
            while buf:
                lines += buf.count(b"\n")
                buf = read_f(buf_size)

        return lines

    @staticmethod
    def name(path):
        return str(path).replace("/", ".").replace(".py", "")


def get_pyvis_options():
    options = """{
        "edges": {
            "arrows": {
            "to": {
                "enabled": true,
                "scaleFactor": 0.3
            },
            "middle": {
                "enabled": true,
                "scaleFactor": 0.3
            }
            },
            "color": {
            "inherit": true
            },
            "smooth": {
                "type": "continuous",
                "forceDirection": "none"
            }
        },
        "nodes": {
            "color": {
                "border": "rgba(100,100,100,0.7)",
                "background": "rgba(150,150,150,1)",
                "highlight": {
                    "border": "blue",
                    "background": "rgba(80,80,255,1)"
                },
                "inherit": "true"
            },
            "font": {
                "size": 12
            },
            "scaling": {
                "min": 12
            },
            "shapeProperties": {
                "borderRadius": 4
            }
        },
        "physics": {
            "forceAtlas2Based": {
                "springLength": 100
            },
            "minVelocity": 0.75,
            "solver": "forceAtlas2Based",
            "timestep": 0.8
        }
    }"""
    return f"""var options = {options}"""


def plot_network(path, saveas=None, **kwargs):
    """
    Maps `.py` files in a folder into a network based on cross-imports and plots it.
    Automatically highlights circular imports, and ignores folders specified in
    `.gitignore`

    Parameters:
    ---
    path: Path. Folder to map

    saveas: str. Html file for the network representation

    **kwargs: Keyword arguments passed to `FileNetwork()`
    """
    if saveas is None:
        saveas = "_srcnetwork.html"
    fn = FileNetwork(path, **kwargs)
    nt = Network("1500px", "1500px")
    nt.toggle_physics(True)
    nt.from_nx(fn.network)
    nt.set_options(get_pyvis_options())
    nt.show(f"{saveas}")
    return


@click.command(
    help="""
    Maps python files in a folder into a network based on cross-imports and plots it.
    Automatically highlights circular imports, and ignores folders specified in
    .gitignore
    """
)
@click.argument("path", type=click.Path(exists=True))
@click.option(
    "--saveas", default="_scrnetwork.html", show_default=True, type=str,
    help="html file where to save the network plot."
)
def plotnetwork(path, saveas):
    plot_network(path, saveas=saveas)


if __name__ == "__main__":
    plotnetwork()
