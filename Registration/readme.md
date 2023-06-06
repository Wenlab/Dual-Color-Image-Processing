
For Registration, do this in the following order under script path:

1. regist_atlas.sh

2. template_run.m

3. regist_mean.sh

4. demonsRegist_run.m

5. interp_run.m

Or run the pipeline_demo.sh under script path.

bash pipeline_demo.sh

Then you will get some directories under data/G/regist_green and data/R/regist_red.

1. red/green_crop --- contain eyes cropped red/green image in mat format.

2. red/green_crop_MIPs --- contain the maximum intensity projections in three directions of the eyes cropped images.

3. red/green_demons --- contain demon registed red/green images in mat format.

4. red/green_demons_MIPs --- contain the maximum intensity projections in three directions of the demon registed images.
