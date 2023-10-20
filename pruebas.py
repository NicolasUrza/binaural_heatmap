
import numpy as np
with open('/home/nicolas/Escritorio/CINTRA/Headpose_Estimation/annotations/01/frame_00004_pose.bin', 'rb') as fid:  # opening the ground truth file
    data_array = np.fromfile(fid, np.float32)
    print(str(data_array))
    para = data_array[3:]
    print(str(para))