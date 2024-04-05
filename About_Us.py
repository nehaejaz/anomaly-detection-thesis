import streamlit as st
import pandas as pd
import numpy as np

st.title('A Few-Shot Learning Method for Single-Object Visual Anomaly Detection')
st.header('Abstract')
st.write("""We propose a few-shot learning method for visually inspecting single objects in an industrial
setting. The proposed method is able to identify whether or not an object is defective
by comparing its visual appearance with a small set of images of the “working” object,
i.e., the object that passes the visual inspection. The method does not require images
of defective objects. Furthermore, the method does not need to be “trained” when used
to inspect new, previously unseen, objects. This suggests that the method can be easily
deployed in industrial settings. We have evaluated the method on three visual anomaly
detection benchmarks—1) MVTec, 2) MPDD, and 3) VisA—and the proposed method
achieves performance that is comparable to state-of-the-art methods that require access
to object-specific training data. Our method also boasts fast inference times, which is
a plus for industry applications. This project is funded in part by Axiom Plastics Inc.,
and we have evaluated the proposed method on a proprietary dataset provided by Axiom.
The results confirm that the proposed method is well-suited for single-object visual
anomaly detection in industry settings.""")
st.header("About the Author")
st.write("")
