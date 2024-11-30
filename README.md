# A fun mini-project
This is a fun mini project that I created after an experience in a strange big city that nearly left me homeless for a night, because I was stuck in an area which was jam-packed with traffic and had no way to reach any Uber that I was booking, even though I was able to walk several blocks if needed, and the Uber had no easy way to get to me without going through traffic. I figured that a feature should exist on ride-hailing apps that optimizes pick-up point selection if asked to (at the time, you had to necessarily select the pick up point) based on current traffic trends. I tried my hand at this problem using a simplified model of streets and traffic and solved the problem using discrete Upper-Confidence-Bound (UCB) optimization of total time taken to get to the ride + trip time.

<p align='center'>
<img src=images/uberplot.png width="800">
</p>

The figure from [1] shows the effects of a spike in demand (in big cities like SF and NYC, usually accompanied by traffic jams near event venues) on ride completion rates; they usually plummet. Surge pricing was not shown to mitigate this, leading to the hypothesis that the issue may not just be in the economics of the ride-hailing model, but also the accessibility of cabs secured through the app.

The detailed solution approach (involving Dijkstra's SP algorithm) is outlined in the write-up "AA222_FinalPaper.pdf". Some results are presented here:

<p align='center'>
<img src=images/case1_mov.png width="400">
<img src=images/case1_nomov.png width="400">
</p>

The plots show cases where a customer would be static and the cab arrive at their location, versus a case where the customer is able to walk at a given speed and the algorithm decided a common meet-up location for the cab and customer. The latter solution clearly allows the cab to avoid certain dense-traffic routes. This solution is impactful if the customer is in an area with high-traffic density

<p align='center'>
<img src=images/case3_mov.png width="400">
<img src=images/case3_nomov.png width="400">
</p>

Another example showing the usefulness of including customer mobility in the route planning process for ride-hailing apps.

References:
[1] Hall, J., Kendrick, C., and Nosko, C., “The Effects of Uber’s Surge Pricing: A Case Study,” Uber Research, 2015.
