# Prepare your AFM data
This instruction aims to help you prepare the data suitable for programs of 3D surface reconstruction.

After finishing previous multi-view AFM scans, you get nine AFM data (**.spm**) with the same resolution, including one overhead scan and eight tilt scans.

Use [Gwyddion](http://gwyddion.net/download.php) to convert the raw AFM scan data (**.spm**) into a matrix (**.txt**). For example, when you do an AFM scan with 256 lines of 256 points, you must get the original text matrix consisting of 256 rows of 256 numbers in meters each.

Put these matrix files into a folder, and name them 00.txt, 01.txt..., 08.txt, where 00.txt must be the conventional overhead view AFM scanning result. There is no requirement on how to sort the remaining eight tilt views.

```
└── Your save folder
    ├── 00.txt (conventional overhead scan)
    ├── 01.txt (tilt scan1)
    ...
    ├── 07.txt (tilt scan7)
    └── 08.txt (tilt scan8) 
```
The scan number and the resolution are modifiable. The scan number defaults to 9 in our experiments and can be modified by changing the **data_num** parameter in our programs. The resolution defaults to 256 x 256, and you can easily achieve what you need by modifying our code. 

### Environment
First, you must install the dependent environment of this preprocessing code.
```
pip install numpy
pip install pillow
pip install tk
pip install opencv
```

### Run Code

This preprocessing code converts the data matrix in **.txt** format to numpy's array file format (**.npz**), initializes each frame's pose, and converts each frame's AFM height value to a virtual orthogonal camera depth value.

Since the scanning range of AFM is very wide, from a few nanometers to tens of micrometers, all AFM data will be transformed to the same scale range during preprocessing so that later programs can use the same parameters and pipeline for 3D reconstruction of AFM data with different scanning ranges. You can provide the scanning range of your data by changing the parameter **scan_range** in micrometers.
For example, the **scan_range** is 10.0 for our TPL structures and 1.5 for our ZIF-67 nanocrystals. 

```
python mark_img.py --input_folder '/Your/save/folder' --output_folder '/Your/output/folder' --scan_range 10.0 --data_num 9
```
While the program is running, it will display all the AFM images in sequence, and the user needs to coarsely click on the approximately same three points in each image.
The ICP is an optimization algorithm that relies on initial values. These user point pairs provide initial poses for the ICP algorithm.
There are many other methods to provide initial poses. For example, you can record the direction of the turntable of each AFM image to get an approximate pose.

Finally, put the whole **output_folder** into **load/** and finish the data preparation. 
