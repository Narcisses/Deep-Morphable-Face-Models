#version 120

varying vec3 vPosition;
varying vec3 vNormal;

const vec3 lightPos = vec3(2.0, 3.0, 4.0);
const vec3 lightColor = vec3(1.0, 0.9686, 0.8549);

void main()
{
    vec3 normal = normalize(vNormal);
    vec3 lightDir = normalize(lightPos - vPosition);
    float ambient = 0.1;
    float diffuse = max(dot(normal, lightDir), 0.0);
    vec3 color = lightColor * diffuse + vec3(ambient);
    gl_FragColor = vec4(color, 1.0);
}
