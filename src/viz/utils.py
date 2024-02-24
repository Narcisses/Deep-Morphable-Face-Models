import numpy as np


def get_projection_matrix(aspect: float, fov: float=45.0, near: float=0.01, far: float=100.0) -> np.ndarray:
    '''Get projection matrix for OpenGL.'''
    
    f = 1.0 / np.tan(np.radians(fov) / 2.0)
    projection_matrix = np.array([
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, (far + near) / (near - far), -1.0],
        [0.0, 0.0, (2.0 * far * near) / (near - far), 0.0]
    ], dtype=np.float32)

    return projection_matrix


def calculate_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    '''Get vertex normals for mesh. Return array of normals for each vertex.'''

    normals = np.zeros(vertices.shape, dtype=vertices.dtype)

    for face in faces:
        v1 = vertices[face[0]]
        v2 = vertices[face[1]]
        v3 = vertices[face[2]]
        normal = np.cross(v1 - v2, v1 - v3)
        normal /= np.linalg.norm(normal)
        normals[face[0]] += normal
        normals[face[1]] += normal
        normals[face[2]] += normal

    normals /= np.linalg.norm(normals, axis=1)[:, None]

    return normals


def get_flat_vertices(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    '''Flatten vertices and return 1-d array of vertices.'''

    return np.array([vertices[vertex] for face in faces for vertex in face], dtype=np.float32)


def get_flat_normals(normals: np.ndarray, faces: np.ndarray) -> np.ndarray:
    '''Flatten vertices and return 1-d array of normals.'''

    return np.array([normals[vertex] for face in faces for vertex in face], dtype=np.float32)


def concat_vertices(flat_vertices: np.ndarray, flat_normals: np.ndarray) -> np.ndarray:
    '''Concatenate vertices and normals (x, y, z, nx, ny, nz) to prepare for drawing.'''

    return np.concatenate((flat_vertices, flat_normals), axis=1)
