import numpy as np

import argparse

import open3d as o3d

POINT_CLOUD_FORMATS = ['npy', 'csv', 'pcd']


def convertPointcloud(inputPath, outputPath):
    """
    Convert a pointcloud file from from any of the
    following formats to any other format:

        - Numpy binary ("npy")
        - Text file ("csv")
        - Pointcloud file ("pcd")

    Parameters
    ----------
    inputFile : str
        Path to the input file.

    outputFile : str
        Path to the output file. Output format will be inferred
        from the output file extension.
    """
    inputExt = inputPath.split('.')[-1].lower()
    outputExt = outputPath.split('.')[-1].lower()

    if inputExt == outputExt:
        raise Exception('Input format is the same as output extension.')

    print(f'Converting from {inputExt} to {outputExt}')

    # Read in data
    if inputExt == 'csv':
        rawData = np.genfromtxt(inputPath, delimiter=',')
        
    elif inputExt == 'npy':
        with open(inputPath, 'rb') as f:
            rawData = np.load(f)

    elif inputExt == 'pcd':
        rawData = np.asarray(o3d.io.read_point_cloud(inputPath).points)
    
    else:
        raise Exception(f'Unsupported input format: {inputExt}')

    # Write to new format
    if outputExt == 'csv':
        np.savetxt(outputPath, rawData)

    elif outputExt == 'npy':
        with open(outputPath, 'wb') as f:
            np.save(f, rawData)

    elif outputExt == 'pcd':
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(rawData)

        o3d.io.write_point_cloud(outputPath, pcd)


if __name__ == '__main__':
    # This can be run as a command line script too
    parser = argparse.ArgumentParser()

    parser.add_argument(dest='inputPath', type=str)
    parser.add_argument(dest='outputPath', type=str)

    args = parser.parse_args()
   
    convertPointcloud(args.inputPath, args.outputPath)
