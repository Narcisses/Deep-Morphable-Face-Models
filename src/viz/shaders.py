import sys
from OpenGL import GL as gl


def load_shaders(vertex_shader_path: str, fragment_shader_path: str) -> int:
    '''
    Load shaders from files and compile them.
    Return shader program ID.
    '''

    vertex_shader_source = open(vertex_shader_path, 'r').read()
    fragment_shader_source = open(fragment_shader_path, 'r').read()

    vertex_shader = gl.glCreateShader(gl.GL_VERTEX_SHADER)
    gl.glShaderSource(vertex_shader, vertex_shader_source)
    gl.glCompileShader(vertex_shader)
    check_shader_compile(vertex_shader)

    fragment_shader = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)
    gl.glShaderSource(fragment_shader, fragment_shader_source)
    gl.glCompileShader(fragment_shader)
    check_shader_compile(fragment_shader)

    shader_program = gl.glCreateProgram()
    gl.glAttachShader(shader_program, vertex_shader)
    gl.glAttachShader(shader_program, fragment_shader)
    gl.glLinkProgram(shader_program)
    check_program_link(shader_program)

    gl.glDeleteShader(vertex_shader)
    gl.glDeleteShader(fragment_shader)

    return shader_program


def check_shader_compile(shader) -> None:
    success = gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS)
    if not success:
        info_log = gl.glGetShaderInfoLog(shader)
        print(f'Shader compilation error:\n{info_log.decode("utf-8")}')
        sys.exit(1)


def check_program_link(program) -> None:
    success = gl.glGetProgramiv(program, gl.GL_LINK_STATUS)
    if not success:
        info_log = gl.glGetProgramInfoLog(program)
        print(f'Program linking error:\n{info_log.decode("utf-8")}')
        sys.exit(1)


def get_shader_variables(shader_program) -> tuple:
    ''' Return attribute and uniform locations.'''

    vertex_position_location = gl.glGetAttribLocation(
        shader_program, 'position')
    vertex_normal_location = gl.glGetAttribLocation(shader_program, 'normal')
    projection_location = gl.glGetUniformLocation(shader_program, 'projection')

    return vertex_position_location, vertex_normal_location, projection_location


def delete_shader_program(shader_program) -> None:
    '''Clear shader program.'''

    gl.glDeleteProgram(shader_program)


def delete_vbo(vbo) -> None:
    '''Clear VBO.'''

    gl.glDeleteBuffers(1, [vbo])
