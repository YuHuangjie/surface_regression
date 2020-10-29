import json
import numpy as np

# parameters
radius = 4.0
nsamples = 800
tag = 'train'
output_file = 'transforms_train.json'

# Generate samples uniforms on a unit hemisphere
samples = np.zeros((nsamples, 3))
for i in range(nsamples):
        x = np.random.normal(0, 1)
        y = np.random.normal(0, 1)
        z = np.random.normal(0, 1)
        z = z if z > 0 else -z
        norm = (x*x+y*y+z*z)**(0.5)
        x, y, z = (x/norm*radius, y/norm*radius, z/norm*radius)
        samples[i, :] = [x, y, z]

# final output
output = {
        'camera_angle_x': 0.6911112070083618,
}
frames = []

# Consider each sample as the COP of a virtual camera, and recover its RT
# matrices
for i, sample in enumerate(samples):
        frame = {
                'file_path': f'./{tag}/r_{i}',
                'rotation': 0.012566370614359171,
        }
        zaxis = sample / np.linalg.norm(sample)
        xaxis = np.cross(np.array([0, 0, 1]), zaxis)
        xaxis /= np.linalg.norm(xaxis)
        yaxis = np.cross(zaxis, xaxis)
        yaxis /= np.linalg.norm(yaxis)

        frame['transform_matrix'] = [
                [
                        xaxis[0],
                        yaxis[0],
                        zaxis[0],
                        sample[0]
                ],
                [
                        xaxis[1],
                        yaxis[1],
                        zaxis[1],
                        sample[1]
                ],
                [
                        xaxis[2],
                        yaxis[2],
                        zaxis[2],
                        sample[2]
                ],
                [
                        0.0,
                        0.0,
                        0.0,
                        1.0
                ]
        ]
        frames.append(frame)

output['frames'] = frames

with open(output_file, 'w') as f:
        f.write(json.dumps(output, indent=4))