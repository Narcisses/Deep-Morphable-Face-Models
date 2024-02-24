import sys
import torch
import ctypes
import numpy as np
from OpenGL import GL as gl
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtWidgets import QApplication, QPushButton

from training.meshio import load_neutral_mesh
from training.model import load_model
from viz.utils import *
from viz.shaders import *


# Constants
WINDOW_SIZE = 600
UPDATE_DELAY = 16
TITLE = 'Interpolation Demo'

MODEL = 'model103.pt'
REFERENCE_NEUTRAL = 'reference_neutral.pt'
VERTEX_SHADER = 'src/viz/shader/vertex_shader.glsl'
FRAGMENT_SHADER = 'src/viz/shader/fragment_shader.glsl'


class Window(QOpenGLWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(TITLE)
        self.resize(WINDOW_SIZE, WINDOW_SIZE)

        self.create_UI()

        # Set callback for timer (milliseconds)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(UPDATE_DELAY)

        # Load mesh via trimesh
        self.vertices, self.faces = load_neutral_mesh(REFERENCE_NEUTRAL)
        self.vertices = np.array(self.vertices).astype(np.float32)
        self.faces = np.array(self.faces).astype(np.uint32)
        self.normals = calculate_normals(self.vertices, self.faces)
        self.flat_vertices = get_flat_vertices(self.vertices, self.faces)
        self.flat_normals = get_flat_normals(self.normals, self.faces)

        # Load model
        self.model, self.checkpoint = load_model(MODEL)

        # Latent variables
        self.z_id = torch.randn(1, self.checkpoint['latent_dim_id'])
        self.z_exp = torch.randn(1, self.checkpoint['latent_dim_exp'])

    def create_UI(self):
        # Add button widgets
        base_style = 'font-size: 12px; font-weight: bold; border-radius: 10px;'
        style_idle = 'QPushButton { %s background-color: #DDDDDD }' % base_style
        style_hover = 'QPushButton:hover { %s background-color: #EEEEEE }' % base_style
        style_pressed = 'QPushButton:pressed { %s background-color: #FFFFFF }' % base_style
        style = f'{style_idle} {style_hover} {style_pressed}'

        self.button_id = QPushButton('New Identity')
        self.button_id.clicked.connect(self.new_identity)
        self.button_id.setStyleSheet(style)
        self.button_id.setFixedWidth(120)
        self.button_id.setFixedHeight(30)
        self.button_id.setParent(self)
        self.button_id.move(self.window().width() // 2 -
                            self.button_id.width() - 20, 560)

        self.button_exp = QPushButton('New Expression')
        self.button_exp.clicked.connect(self.new_expression)
        self.button_exp.setStyleSheet(style)
        self.button_exp.setFixedWidth(120)
        self.button_exp.setFixedHeight(30)
        self.button_exp.setParent(self)
        self.button_exp.move(self.window().width() // 2 + 20, 560)

    def closeEvent(self, event) -> None:
        '''Close window event.'''

        delete_shader_program(self.shader_program)
        delete_vbo(self.vbo)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        '''
        Listen to key event.
        Take screenshot if S key pressed.
        '''

        if event.key() == Qt.Key.Key_S:
            screen = QApplication.primaryScreen()
            shot = screen.grabWindow(w.winId())
            shot.setDevicePixelRatio(2)
            shot.save('screenshot.png', 'png', 100)

    def initializeGL(self) -> None:
        '''Initialize OpenGL (create context).'''

        gl.glClearColor(0.5, 0.5, 0.5, 1)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDisable(gl.GL_CULL_FACE)

        # Concatenate vertices and normals (x, y, z, nx, ny, nz)
        # Use these data to draw the mesh
        self.flat_vertices = concat_vertices(
            self.flat_vertices, self.flat_normals)

        self.vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.flat_vertices.nbytes,
                        self.flat_vertices, gl.GL_DYNAMIC_DRAW)

        # Load shaders
        self.shader_program = load_shaders(VERTEX_SHADER, FRAGMENT_SHADER)

        # Get attribute and uniform locations
        self.vertex_position_location, self.vertex_normal_location, \
            self.projection_location = get_shader_variables(
                self.shader_program)
        
        # Update new person
        self.update_mesh()

    def resizeGL(self, width, height) -> None:
        '''
        Resize OpenGL viewport.
        And set up projection matrix.
        '''

        gl.glViewport(0, 0, width, height)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        aspect = width / float(height)

        self.projection_matrix = get_projection_matrix(aspect)

        gl.glLoadMatrixf(self.projection_matrix)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()

    def paintGL(self) -> None:
        '''Draw mesh.'''

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glLoadIdentity()

        # Use the shader program
        gl.glUseProgram(self.shader_program)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)

        gl.glVertexAttribPointer(
            self.vertex_position_location, 3, gl.GL_FLOAT, False, 4 * 6, None)
        gl.glEnableVertexAttribArray(self.vertex_position_location)
        gl.glVertexAttribPointer(
            self.vertex_normal_location, 3, gl.GL_FLOAT, False, 4 * 6, ctypes.c_void_p(3 * 4))
        gl.glEnableVertexAttribArray(self.vertex_normal_location)

        gl.glUniformMatrix4fv(self.projection_location, 1,
                              gl.GL_FALSE, self.projection_matrix)

        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
        gl.glEnableClientState(gl.GL_NORMAL_ARRAY)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, len(self.flat_vertices))
        gl.glDisableClientState(gl.GL_NORMAL_ARRAY)
        gl.glDisableClientState(gl.GL_VERTEX_ARRAY)

    def update_mesh(self) -> None:
        '''Callback for slider widget.'''

        # Decode
        x = self.model.decoder(torch.cat((self.z_id, self.z_exp), dim=1))
        
        # Convert to numpy array
        x = x.view(-1, 3).detach().numpy().astype(np.float32)

        # Add neutral reference
        x += self.vertices

        # Update VBO data
        self.flat_vertices = get_flat_vertices(x, self.faces)
        self.flat_vertices = concat_vertices(
            self.flat_vertices, self.flat_normals)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0,
                           self.flat_vertices.nbytes, self.flat_vertices)

    def new_identity(self) -> None:
        self.z_id = torch.randn(1, self.checkpoint['latent_dim_id'])
        self.update_mesh()

    def new_expression(self) -> None:
        self.z_exp = torch.randn(1, self.checkpoint['latent_dim_exp'])
        self.update_mesh()


if __name__ == "__main__":
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseDesktopOpenGL)
    app = QApplication(sys.argv)
    w = Window()
    w.show()
    sys.exit(app.exec())
