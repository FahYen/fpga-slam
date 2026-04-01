# SG-SLAM Docker Commands (macOS M1, ROS1 Noetic)

## 1) Fast repeat workflow

Most day-to-day runs should look like this:

1. Build the image once, or rebuild it only after changing `Dockerfile`, `ros/`, or `cpp/`.
2. Reuse the same image tag: `sg-slam:noetic`.
3. Reuse the named Docker volumes for `/opt/catkin_ws/build`, `/opt/catkin_ws/devel`, and `/opt/catkin_ws/logs`.
4. Re-run `catkin build` inside the container only after source changes.

If the image already exists and the named volumes are already populated, skip straight to step 4.

## 2) Build the image only when needed

This now prebuilds the catkin workspace once during `docker build`, so container startup does not have to begin from a clean compile every time.

```bash
docker build --platform linux/amd64 --build-arg CATKIN_JOBS=2 -t sg-slam:noetic .
```

Docker will automatically reuse cached image layers on later builds as long as you keep the same Dockerfile flow and do not pass `--no-cache`.

## 3) Pull large files after clone

```bash
git lfs pull
```

## 4) Start the container from the repo root

Use named volumes for the catkin build outputs so they survive `--rm` container cleanup.

```bash
docker run --rm -it --platform linux/amd64 \
  -v "$(pwd):/opt/catkin_ws/src/SG-SLAM" \
  -v sg_slam_catkin_build:/opt/catkin_ws/build \
  -v sg_slam_catkin_devel:/opt/catkin_ws/devel \
  -v sg_slam_catkin_logs:/opt/catkin_ws/logs \
  sg-slam:noetic
```

The first run seeds those named volumes from the image. Later runs reuse them, so startup is much faster and `catkin build` can stay incremental.

## 5) Rebuild only after source changes

The image already configures catkin for a release build, prefers system Eigen/TBB, keeps the bundled newer Ceres that this repo expects, and disables milestone benchmarks. Run this only when you have changed code in the bind-mounted repo.

```bash
source /opt/ros/noetic/setup.bash
cd /opt/catkin_ws
catkin build --no-status -j2 -p1 semgraph_slam
source devel/setup.bash
```

You do not need to rebuild the Docker image for ordinary source edits, because the repo is bind-mounted into the container.

## 6) Launch SG-SLAM

```bash
roslaunch semgraph_slam semgraph_slam_kitti.launch
```

Other launch files:

```bash
roslaunch semgraph_slam semgraph_slam_mulran.launch
roslaunch semgraph_slam semgraph_slam_apollo.launch
```

## 7) When to rebuild the image

Re-run `docker build` when one of these changes:

- `Dockerfile`
- apt/pip/ROS dependency setup
- catkin prebuild settings such as `CATKIN_JOBS`
- anything under `ros/` or `cpp/` that you want baked into a fresh image seed

Do not rebuild the image just because you edited code and want to run it again. In the normal loop, restart the container and run `catkin build` against the cached named volumes.

## 8) Notes

- The image still uses `linux/amd64` for ROS1 Noetic compatibility on Apple Silicon.
- Install Git LFS locally before cloning or pulling large files: `brew install git-lfs`.
- Run the command above from the `SG-SLAM` repository root so `$(pwd)` mounts the repo into the container.
- The Docker image preconfigures catkin with `Release`, `USE_SYSTEM_EIGEN3=ON`, `USE_SYSTEM_TBB=ON`, and `ENABLE_MILESTONE1_BENCHMARKS=OFF`.
- The image intentionally keeps the bundled Ceres build because the Ubuntu 20.04 package is too old for the `ceres::Manifold` API used by this repo.
- The first container run seeds the named volumes from the image's prebuilt workspace; later runs reuse `build`, `devel`, and `logs` so you avoid repeated clean rebuilds.
- Increase `CATKIN_JOBS` during `docker build` and the runtime `catkin build -j...` value only if Docker Desktop has enough CPU and RAM.
- If you need a clean rebuild, remove the named volumes:

```bash
docker volume rm sg_slam_catkin_build sg_slam_catkin_devel sg_slam_catkin_logs
```

- If you also want to force Docker to rebuild image layers from scratch, use:

```bash
docker build --no-cache --platform linux/amd64 --build-arg CATKIN_JOBS=2 -t sg-slam:noetic .
```

- `data/kitti/sequences/00` is intended to be shared through Git LFS for a reproducible KITTI sequence 00 setup.
- Replace dataset paths in launch files (`lidar_path`, `label_path`) with your local paths.
