# eval depthtrack
cd Depthtrack_workspace
vot evaluate --workspace ./ vipt_deep
vot analysis --nocache --name vipt_deep
cd ..

# eval vot22-rgbd
cd vot22_RGBD_workspace
vot evaluate --workspace ./ vipt_deep
vot analysis --nocache --name vipt_deep
cd ..

