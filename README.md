# disc-metrics
This program uses computer vision to automatically calculate speed, angle, and body position using nothing but a video of a throw. Speed is directly correlated with distance, one of the most sought after skills in disc golf, so having an accurate way to measure speed is super helpful in nailing down what works and what doesn't when making form changes. However, other than expensive equipment such as a radar gun or TechDisc, there really don't exist any tools to easily measure the speed of a throw.

## Determine distance from camera automatically via known radius of disc
![](https://github.com/DiscMetrics/disc-metrics/blob/examples/data/disc_metrics_circles-cropped-2.gif)

## Separate the disc from the background
![](https://github.com/DiscMetrics/disc-metrics/blob/examples/data/disc_metrics_threshold-cropped.gif)

## Calculate bounding box of the flying disc, getting position data
![](https://github.com/DiscMetrics/disc-metrics/blob/examples/data/disc_metrics_disc_flying-cropped.gif)

## Track form using AI model, gathering 3D position data
![](https://github.com/DiscMetrics/disc-metrics/blob/examples/data/disc_metrics_overlay-cropped.gif)

## Visualize throwing form from any angle
![](https://github.com/DiscMetrics/disc-metrics/blob/examples/data/disc_metrics_wireframe-cropped.gif)


usage: `python3 process.py path_to_video`
* `python3 process.py --help` for help and list of optional arguments 
