import streamlit as st

import streamlit as st
import pandas as pd
import numpy as np

st.write("""
# My first app
Hello *world!*
""")
np.random.seed(12345)

dataframe = pd.DataFrame(
    np.random.randn(10, 20),
    columns=('col %d' % i for i in range(20)))

st.dataframe(dataframe.style.highlight_max(axis=0))

st.write("""
## Line chart
In this subsection, we will test `line chart` after creating a dataframe. 
""")
chart_data = pd.DataFrame(
     np.random.randn(20, 3),
     columns=['a', 'b', 'c'])

st.line_chart(chart_data)

st.write("""
## Map
In this subsection, we will test `Map` with data after creating a dataframe. 
""")
map_data = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon'])

st.map(map_data)

if st.checkbox('Show dataframe'):
    chart_data = pd.DataFrame(
       np.random.randn(20, 3),
       columns=['a', 'b', 'c'])

    chart_data


# Add a selectbox to the sidebar:
add_selectbox = st.sidebar.selectbox(
    'How would you like to be contacted?',
    ('Email', 'Home phone', 'Mobile phone')
)

# Add a slider to the sidebar:
add_slider = st.sidebar.slider(
    'Select a range of values',
    0.0, 100.0, (25.0, 75.0)
)  

left_column, right_column = st.columns(2)
# You can use a column just like st.sidebar:
left_column.button('Press me!')

# Or even better, call Streamlit functions inside a "with" block:
with right_column:
    chosen = st.radio(
        'Sorting hat',
        ("Gryffindor", "Ravenclaw", "Hufflepuff", "Slytherin"))
    st.write(f"You are in {chosen} house!")
    


if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame(np.random.randn(20, 2), columns=["x", "y"])

st.header("Choose a datapoint color")
color = st.color_picker("Color", "#FF0000")
st.divider()
st.scatter_chart(st.session_state.df, x="x", y="y", color=color)

from PIL import Image

# Load image (can be .png, .jpg, .webp, etc.)
img = Image.open("/users/cmk2000/sharedscratch/Datasets/LSUN/test/test_0000001.png")

# Display it in Streamlit
st.image(img, caption="My Image", use_column_width=True)



import matplotlib.pyplot as plt


import io
from pathlib import Path

# Load multiple images from a folder
image_paths = list(Path("/users/cmk2000/sharedscratch/Datasets/LSUN/test").rglob("*.png"))[:4]  # Adjust number as needed
images = [Image.open(p) for p in image_paths]

# Create a subplot grid (2x2 in this case)
fig, axs = plt.subplots(2, 2, figsize=(8, 8))

for ax, img, path in zip(axs.flatten(), images, image_paths):
    ax.imshow(img)
    ax.set_title(path.name)
    ax.axis('off')

# Save to a buffer
buf = io.BytesIO()
plt.tight_layout()
plt.savefig(buf, format="png")
buf.seek(0)

# Display in Streamlit
st.image(buf, caption="Subplot of Images", use_column_width=True)


st.write(
    """
    ###  Additional test.
    
    """
)

if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame(np.random.randn(20, 2), columns=["x", "y"])

st.header("Choose a datapoint color")
color = st.color_picker("Color", "#FF0000")
st.divider()
st.scatter_chart(st.session_state.df, x="x", y="y", color=color)