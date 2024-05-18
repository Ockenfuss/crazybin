import streamlit as st
import matplotlib.pyplot as plt
import math
from crazybin import imshow

def main():
    st.title("Image Tesselation")
    st.write("This app allows you to transform your images into a parquet of small tiles.")
    st.write("Just upload and image and select the type of tile you want together with the resolution.")

    tile_descriptions = {
            "Regular hexagon": "hex",
            "Composition of a regular hexagon, triangles and squares": "hex_rhomb",
            "Composition of three lizard shaped tiles inspired by M.C. Escher": "reptile",
            "Composition of four frog shaped tiles inspired by M.C. Escher.": "frog", 
            "Irregular P3 penrose tiling, consisting of two rhombs with different angles.": "pen_rhomb",
        }
    tile_types={"hex": "regular", "hex_rhomb": "regular", "reptile": "regular", "frog": "regular", "pen_rhomb": "irregular"} 
    
    file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])
    selected_description = st.selectbox("Choose a tile type", list(tile_descriptions.keys()))
    tile_key=tile_descriptions[selected_description]
    slidermax=10
    resolution = st.slider("Choose the resolution", min_value=1, max_value=slidermax, value=5)

    if file is not None:
        image=plt.imread(file)/255
        st.subheader("Original Image:")
        st.image(image, use_column_width=True)

        if tile_types[tile_key]=="regular":
            #go exponentially from 1 to 100
            resolution=math.ceil(100**((resolution-1)/(slidermax-1)))


        st.subheader("Tesselated image:")
        print(resolution)
        print(tile_types[tile_key])
        fig, ax = plt.subplots()
        imshow(image, tile=tile_key, ax=ax, gridsize=resolution)
        ax.axis('off')
        st.pyplot(fig)

if __name__ == "__main__":
    main()