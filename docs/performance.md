---
title: Performance
---

<!-- ## Theoretical Speedup

We use Gustafson's Law to calculate speedup: 

$$S=p+(1-p)s$$

Running 8 p2.xlarge instances as worker nodes gives $$p=8$$. We can then calculate the speedup for the two steps of our calculations.

<ol>
    <li>Create bounding boxes with NNs (fully parallelizable, store boxes in s3)</li>
    $$S = 8 + (-7)0 = 8$$
    <li>Compute analytics from bounding boxes in Spark: 8 workers, 2 cores per worker, 2 threads per core</li>
    $$S=32$$

</ol>

TODO: Talk about performance -->

<img src="{{ "images/plots1.png" | relative_url}}" >
<img src="{{ "images/plots2.png" | relative_url}}" >


<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>