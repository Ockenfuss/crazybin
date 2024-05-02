import numpy as np
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from shapely import affinity
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import json
from importlib import resources
import cmath
from pathlib import Path


GOLDEN = (1 + np.sqrt(5)) / 2

class View(object):
    def __init__(self, xmin, xmax, ymin, ymax):
        if xmin>xmax:
            raise ValueError('xmin must be smaller than xmax')
        if ymin>ymax:
            raise ValueError('ymin must be smaller than ymax')
        self.xmin=xmin
        self.xmax=xmax
        self.ymin=ymin
        self.ymax=ymax
        self.width=self.xmax-self.xmin
        self.height=self.ymax-self.ymin
        self.center=[(self.xmin+self.xmax)/2, (self.ymin+self.ymax)/2]
        self.diagonal=np.sqrt((self.xmax-self.xmin)**2+(self.ymax-self.ymin)**2)
    
    def __repr__(self) -> str:
        return f'View: {self.xmin:.2f}, {self.xmax:.2f}, {self.ymin:.2f}, {self.ymax:.2f}'

    def check_point_in_view(self, x,y):
        return (x>=self.xmin) & (x<=self.xmax) & (y>=self.ymin) & (y<=self.ymax)
    
    def get_corners(self):
        return ((self.xmin, self.ymin), (self.xmax, self.ymin), (self.xmax, self.ymax), (self.xmin, self.ymax))
    
    def check_corner_in(self, other):
        for x,y in self.get_corners():
            if other.check_point_in_view(x,y):
                return True
        return False

    

    def pad(self, tile):
        box=tile.get_bounding_box()
        view_padded=View(self.xmin-box.width, self.xmax+box.width, self.ymin-box.height, self.ymax+box.height)
        return view_padded
    
    def plot(self, ax=None, **kwargs):
        if ax is None:
            fig, ax=plt.subplots()
        ax.plot([self.xmin, self.xmax, self.xmax, self.xmin, self.xmin], [self.ymin, self.ymin, self.ymax, self.ymax, self.ymin], **kwargs)



class Grid(object):
    def __init__(self, g1, g2):
        self.g1=g1
        self.g2=g2
        self.total_dx=abs(g1[0])+abs(g2[0]) #sum of x components
        self.total_dy=abs(g1[1])+abs(g2[1])

    @classmethod
    def from_json(cls, path):
        with open(path, 'r') as f:
            json_dict=json.load(f)
            grid=Grid(json_dict['v1'], json_dict['v2'])
        return grid
    
    def scale(self, sx, sy):
        g1_scaled=[sx*self.g1[0], sy*self.g1[1]]
        g2_scaled=[sx*self.g2[0], sy*self.g2[1]]
        return Grid(g1_scaled, g2_scaled)
    
    def xy_to_g(self,x,y):
        ind_i=(x*self.g2[1]-y*self.g2[0])/(self.g1[0]*self.g2[1]-self.g1[1]*self.g2[0])
        ind_j=(y*self.g1[0]-x*self.g1[1])/(self.g1[0]*self.g2[1]-self.g1[1]*self.g2[0])
        return ind_i, ind_j

    def g_to_xy(self, i,j):
        x=self.g1[0]*i + self.g2[0]*j
        y=self.g1[1]*i + self.g2[1]*j
        return x,y

    def xy_to_g_int(self, x,y):
        ind_i, ind_j=self.xy_to_g(x,y)
        return np.round(ind_i), np.round(ind_j)

    def __repr__(self) -> str:
        return f'Grid g1: {self.g1[0]:.3f},{self.g1[1]:.3f}; g2: {self.g2[0]:.3f},{self.g2[1]:.3f}'
    
    def plot(self, ax=None, **kwargs):
        if ax is None:
            fig, ax=plt.subplots()
        ax.plot([0, self.g1[0]], [0, self.g1[1]], **kwargs)
        ax.plot([0, self.g2[0]], [0, self.g2[1]], **kwargs)
        # ax.annotate("", xy=(self.g1[0], self.g1[1]), xytext=(0, 0), arrowprops=dict(arrowstyle="->"))
        # ax.annotate("", xy=(self.g2[0], self.g2[1]), xytext=(0, 0), arrowprops=dict(arrowstyle="->"))

class GridView(object):
    def __init__(self, grid, points_i, points_j) -> None:
        self.grid=grid
        self.points_i=points_i
        self.points_j=points_j
    
    @classmethod
    def from_circle(cls, grid, radius):
        """Return: points in g coordinates
        """
        theta=-np.arctan2(grid.g1[1], grid.g1[0])
        R=np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        g1_rotated=np.dot(R, grid.g1)
        g2_rotated=np.dot(R, grid.g2)

        points=[]
        j_start=-np.ceil(radius/abs(g2_rotated[1]))
        j=j_start
        while j<=-j_start:
            if_start=(-radius-j*g2_rotated[0])/g1_rotated[0]
            sign_if=np.sign(if_start)
            i_start=sign_if*np.ceil(abs(if_start))
            if_end=(radius-j*g2_rotated[0])/g1_rotated[0]
            sign_if=np.sign(if_end)
            i_end=sign_if*np.ceil(abs(if_end))
            i=i_start
            while i<=i_end:
                x=g1_rotated[0]*i + g2_rotated[0]*j
                y=g1_rotated[1]*i + g2_rotated[1]*j
                if x**2+y**2<=radius**2:
                    points.append([i,j])
                i+=np.sign(if_end-if_start)
            j+=1
        points=np.array(points)
        return cls(grid, points[:,0], points[:,1])
    
    @classmethod
    def from_view(cls, grid, view):
        center_xy=view.center
        offset_ind=grid.xy_to_g_int(center_xy[0], center_xy[1])
        radius=1.5*view.diagonal
        gridview=cls.from_circle(grid, radius)
        gridview=gridview.translate(*offset_ind)
        return gridview.get_in_view(view)
    
    def translate(self, i:int,j:int):
        return GridView(self.grid, self.points_i+i, self.points_j+j)
    
    def get_xy(self):
        return self.grid.g_to_xy(self.points_i, self.points_j)
    
    def get_in_view(self, view):
        points_x, points_y=self.get_xy()
        in_view=view.check_point_in_view(points_x, points_y)
        return GridView(self.grid, self.points_i[in_view], self.points_j[in_view])

    def plot(self, ax=None, **kwargs):
        if ax is None:
            fig, ax=plt.subplots()
        points_x, points_y=self.grid.g_to_xy(self.points_i, self.points_j)
        ax.scatter(points_x, points_y, **kwargs)

class Tile(object):
    def __init__(self, atoms):
        self.atoms=atoms
        self.natoms=len(atoms)
        self.union=unary_union(atoms)
        self.centroid=self.union.centroid
    
    @classmethod
    def from_json(cls, path):
        with open(path, 'r') as f:
            tile=json.load(f)
            atoms=[Polygon(t) for t in tile['atoms']]
        tile=cls(atoms)
        tile=tile.center()
        return tile
    
    def get_bounding_box(self):
        bounds=self.union.bounds
        return View(xmin=bounds[0], xmax=bounds[2], ymin=bounds[1], ymax=bounds[3])
    
    def center(self):
        atoms_shifted=[affinity.translate(atom, -self.centroid.x, -self.centroid.y) for atom in self.atoms]
        return Tile(atoms_shifted)
    
    def scale(self, sx=1.0, sy=1.0):
        atoms_scaled=[affinity.scale(atom, sx, sy, origin=self.centroid) for atom in self.atoms]
        return Tile(atoms_scaled)
    
    def translate(self, dx=0.0, dy=0.0):
        atoms_translated=[affinity.translate(atom, dx, dy) for atom in self.atoms]
        return Tile(atoms_translated)
    
    def contains(self, point: Point)->int:
        for i, atom in enumerate(self.atoms):
            if atom.contains(point):
                return i
        return -1
    
    def __getitem__(self, key):
        return self.atoms[key]
    
    def __repr__(self):
        box=self.get_bounding_box()
        description=f'Tile with {len(self.atoms)} atoms\nBounding box: {box}'
        return description
    
    def plot(self, ax=None, **kwargs):
        if ax is None:
            fig, ax=plt.subplots()
        for atom in self.atoms:
            x, y = atom.exterior.xy
            ax.fill(x, y, **kwargs)
            # ax.plot(x, y, **kwargs)

class LookupTable(object):
    def __init__(self, tile: Tile, grid:Grid):
        box=tile.get_bounding_box()
        gridview=GridView.from_circle(grid, radius=box.diagonal)
        points_x, points_y=gridview.get_xy()
        distance=np.sqrt(points_x**2+points_y**2)
        ind_sorted=np.argsort(distance)
        self.sorted_neighbours=np.array([gridview.points_i[ind_sorted], gridview.points_j[ind_sorted]]).T

class Parquet(object):
    def __getitem__(self, key):
        return self.tiles[key]

    def plot(self, ax=None, plot_view=True, **kwargs):
        if ax is None:
            fig, ax=plt.subplots()
        for tile in self.tiles.values():
            tile.plot(ax=ax, **kwargs)
        if plot_view:
            self.view.plot(ax=ax, color='black')


class RegularParquet(Parquet):
    def __init__(self, tile: Tile, grid:Grid, view:View, gridsize=10):
        if np.iterable(gridsize):
            nx, ny=gridsize
        elif np.isscalar(gridsize):
            nx=gridsize
            ny=None
        else:
            raise ValueError('gridsize must be a scalar or a tuple')

        sx=view.width/(nx*grid.total_dx)
        if ny is None:
            sy=sx
        else:
            sy=view.height/(ny*grid.total_dy)
        self.root_tile=tile.scale(sx, sy)
        self.grid=grid.scale(sx, sy)
        self.view=view

        view_padded=view.pad(self.root_tile)
        gridview=GridView.from_view(self.grid, view_padded)
        self.tiles={}
        for i,j in zip(gridview.points_i, gridview.points_j):
            self.tiles[i, j]=self.root_tile.translate(*self.grid.g_to_xy(i,j))

class PenroseP3Parquet(Parquet):
    def __init__(self, view: View, generations=1):
        if generations>12:
            raise ValueError("More than 12 inflation generations are not recommended.")
        self.view=view
        self.generations=generations

        initial_radius=self.view.diagonal/2*1.3
        center=self.view.center[0]+self.view.center[1]*1j
        triangles=PenroseP3Parquet._create_initial_wheel(center,initial_radius)
        for i in range(generations):
            triangles=PenroseP3Parquet._subdivide(triangles)
        pairs=PenroseP3Parquet._find_rhombus_pairs(triangles)
        polygons=PenroseP3Parquet._to_polygon_rhombs(triangles, pairs)
        tiles=[Tile([p]) for p in polygons]

        tiles=[t for t in tiles if t.get_bounding_box().check_corner_in(self.view)]
        keys=range(len(tiles))
        self.tiles=dict(zip(keys, tiles))

    @staticmethod
    def _create_initial_wheel(center, radius):
        # Create wheel of type 0 ('red') triangles around the origin
        triangles = []
        for i in range(10):
            B = cmath.rect(radius, (2*i - 1) * np.pi / 10)+center
            C = cmath.rect(radius, (2*i + 1) * np.pi / 10)+center
            if i % 2 == 0:
                B, C = C, B  # Make sure to mirror every second triangle
            triangles.append((0, center, B, C))
        return triangles
    
    @staticmethod
    def _subdivide(triangles):
        result = []
        for color, A, B, C in triangles:
            if color == 0:
                # Subdivide red triangle
                P = A + (B - A) / GOLDEN
                result += [(0, C, P, B), (1, P, C, A)]
            else:
                # Subdivide blue triangle
                Q = B + (A - B) / GOLDEN
                R = B + (C - B) / GOLDEN
                result += [(1, R, C, A), (1, Q, R, B), (0, R, Q, A)]
        return result
    
    @staticmethod
    def _find_rhombus_pairs(triangles):
        """find triangles which share the points B,C: Each pair will form a rhombus"""
        triangles_bc=[(np.round(tri[2],10), np.round(tri[3],10)) for tri in triangles]
        visited=[False]*len(triangles_bc)
        pairs=[]
        for i in range(len(triangles_bc)):
            if visited[i]:
                continue
            for j in range(i+1, len(triangles_bc)):
                if visited[j]:
                    continue
                if triangles_bc[i][0]==triangles_bc[j][0] and triangles_bc[i][1]==triangles_bc[j][1]:
                    visited[j]=True
                    pairs.append((i,j))
                    break
        return pairs

    @staticmethod
    def _to_polygon_rhombs(triangles, pairs):
        polygons = []
        for pair in pairs:
            A=triangles[pair[0]]
            B=triangles[pair[1]]
            polygons.append(Polygon([Point(A[1].real, A[1].imag), Point(A[2].real, A[2].imag), Point(B[1].real, B[1].imag), Point(A[3].real, A[3].imag)]))
        return polygons


class ParquetFactory(object):
    @staticmethod
    def _get_from_json(json_name : str, reader: object):
        path=Path(json_name)
        if path.suffix=='.json':
            if not path.exists():
                raise FileNotFoundError(f"File {path.name} not found")
            return reader.from_json(path)
        elif isinstance(json_name, str):
            traversable=resources.files("crazybin")
            with resources.as_file(traversable) as f:
                filepath=f/"tiles" / f"{json_name}.json"
                if not filepath.exists():
                    raise FileNotFoundError(f"JSON resource for {json_name} not found")
                reader_object=reader.from_json(filepath)
            return reader_object
        else:
            raise ValueError('json_name must be a string')

    @staticmethod
    def _get_tile(tile_name):
        return ParquetFactory._get_from_json(tile_name, Tile)
    
    @staticmethod
    def _get_grid(grid_name):
        return ParquetFactory._get_from_json(grid_name, Grid)

    @classmethod
    def from_keyword(cls, tile_name, allow_penrose=True):
        if tile_name[:3]=='pen' and not allow_penrose:
            raise NotImplementedError("Penrose parquets are not yet implemented for this function.")
        if tile_name=='pen_rhomb':
            return PenroseP3ParquetFactory()
        else:
            grid=ParquetFactory._get_grid(tile_name)
            tile=ParquetFactory._get_tile(tile_name)
            return RegularParquetFactory(tile, grid)
        
    def __call__(self, view, resolution):
        return self.get_parquet(view, resolution)


class RegularParquetFactory(ParquetFactory):
    def __init__(self, tile, grid) -> None:
        self.tile=tile
        self.grid=grid
    
    def get_parquet(self, view, gridsize):
        return RegularParquet(self.tile, self.grid, view, gridsize)

class PenroseP3ParquetFactory(ParquetFactory):
    def __init__(self):
        pass

    def get_parquet(self, view, generations):
        return PenroseP3Parquet(view, generations)
    
class ColorParquet(object):
    def __init__(self, parquet: Parquet, colors: dict):
        self.parquet=parquet
        self.colors=colors
        first=next(iter(colors.values()))
        if np.isscalar(first):
            self.ncolors=1
        else:
            self.ncolors=len(first)

    def _create_colormap(self, cmap_name, vmin, vmax):
        norm = mcolors.Normalize(vmin, vmax)
        cmap = plt.get_cmap(cmap_name)
        def map_value_to_color(value):
            return cmap(norm(value))
        return map_value_to_color

        
    def plot(self, ax, cmap='viridis', vmin=None, vmax=None, edgecolor=None, full=False):
        if self.ncolors==1:
            if vmin is None:
                vmin=min(self.colors.values())
            if vmax is None:
                vmax=max(self.colors.values())
            colormap = self._create_colormap(cmap, vmin, vmax)
        else:
            colormap=lambda x: x
        
        if ax is None:
            fig, ax=plt.subplots()

        for key in self.parquet.tiles:
            for k, atom in enumerate(self.parquet[key].atoms):
                x, y = atom.exterior.xy
                ax.fill(x, y, color=colormap(self.colors[key,k]))
                
        if edgecolor is not None:
            for key in self.parquet.tiles:
                for k, atom in enumerate(self.parquet[key].atoms):
                    x, y = atom.exterior.xy
                    ax.plot(x, y, color=edgecolor)
        if not full:
            ax.set_xlim(self.parquet.view.xmin, self.parquet.view.xmax)
            ax.set_ylim(self.parquet.view.ymin, self.parquet.view.ymax)
        return ax

class Histogram(object):
    def __init__(self, x,y,parquet_factory:ParquetFactory,weights=None, areadensity=False, gridsize=10):
        view=View(min(x), max(x), min(y), max(y))
        self.parquet=parquet_factory(view, gridsize)
        self.lut=LookupTable(self.parquet.root_tile, self.parquet.grid)

        keys=[(key,k) for key,tile in self.parquet.tiles.items() for k in range(tile.natoms)]
        self.hist=dict.fromkeys(keys, 0)

        self._count(x,y,weights, areadensity)
    
    def _get_containing_index(self, x,y):
        offset_i, offset_j=self.parquet.grid.xy_to_g_int(x,y)
        for i,j in self.lut.sorted_neighbours:
                try:
                    k=self.parquet.tiles[i+offset_i,j+offset_j].contains(Point(x,y))
                    if k>=0:
                        return i+offset_i,j+offset_j,k
                    else:
                        continue
                except KeyError: #Tile not found. This can happen at the border of the parquet, where the LUT extends beyond the parquet
                    continue
        raise ValueError('Point not found in parquet')

    def _count(self, x,y, weights=None, areadensity=False):
        if weights is None:
            weights=np.ones(len(x))
        for xx,yy, ww in zip(x,y, weights):
            i,j,k=self._get_containing_index(xx,yy)
            self.hist[(i,j),k]+=ww
        if areadensity:
            for key,atom_key in self.hist:
                    self.hist[key,atom_key]/=self.parquet[key][atom_key].area

    def plot(self, ax=None, cmap='viridis', vmin=None, vmax=None, edgecolor=None):
        self.color_parquet=ColorParquet(self.parquet, self.hist)
        return self.color_parquet.plot(ax, cmap=cmap, vmin=vmin, vmax=vmax, edgecolor=edgecolor)


class TileImage(object):
    def __init__(self, image,parquet_factory:ParquetFactory, gridsize=10, extent=None):
        self.image_view=View(0, image.shape[1], 0, image.shape[0])
        if extent is None:
            self.view=self.image_view
        else:
            self.view=View(*extent)
        self.parquet=parquet_factory(self.view, gridsize)
        keys=[(key,k) for key,tile in self.parquet.tiles.items() for k in range(tile.natoms)]
        if image.ndim==2:
            self.colors=dict.fromkeys(keys, 0)
        elif image.ndim==3:
            self.colors=dict.fromkeys(keys, [0]*image.shape[2])
        else:
            raise ValueError('Image must have 2 or 3 dimensions')

        for key in self.parquet.tiles:
            for k, atom in enumerate(self.parquet[key].atoms):
                ix,iy=self._xy_to_image(*atom.centroid.coords[0])
                self.colors[key,k]=image[iy,ix]
    
    def _xy_to_image(self, x,y):
        ix,iy=round((x-self.view.xmin)/(self.view.width)*self.image_view.width), round((y-self.view.ymin)/(self.view.height)*self.image_view.height)
        ix=max(0, min(self.image_view.width-1, ix))
        iy=max(0, min(self.image_view.height-1, iy))
        return ix,iy
        
    def plot(self, ax=None,cmap='viridis', vmin=None, vmax=None, edgecolor=None, aspect='equal', origin='upper'):
        self.color_parquet=ColorParquet(self.parquet, self.colors)
        ax=self.color_parquet.plot(ax, cmap=cmap, vmin=vmin, vmax=vmax, edgecolor=edgecolor)
        if origin=='upper':
            ax.invert_yaxis()
        elif origin=='lower':
            pass
        else:
            raise ValueError('origin must be upper or lower')
        ax.set_aspect(aspect)
        return ax
