#version 120

attribute vec3 position;
attribute vec3 normal;

uniform mat4 projection;

varying vec3 vPosition;
varying vec3 vNormal;

void main()
{
    vPosition = position;
    vNormal = normal;

    gl_Position = projection * vec4(position - vec3(0, 0.1, 3), 1.0);
}
