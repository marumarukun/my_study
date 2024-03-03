import streamlit as st
import pandas as pd
import numpy as np

st.title('My first app')
st.markdown("""
            # Header 1
            ## Header 2
            ### Header 3
            - Bullet 1
                - Bullet 2
            """)


df = pd.DataFrame(
    np.random.rand(100, 2),
    columns=['a', 'b']
)
st.write(df)

st.line_chart(df)
st.area_chart(df)
st.bar_chart(df)


options = st.multiselect(
    'What are your favorite colors',
    ['Green', 'Yellow', 'Red', 'Blue'],
    ['Yellow', 'Red'])

st.write(f'You selected: {options}')


left_col, right_col = st.columns(2)

with left_col:
    button = st.button('Press me!', type='primary')
    if button:
        st.write('You pressed the button!')

with right_col:
    slide_value = st.slider('Slide me', min_value=0, max_value=10)
    st.write(f'You selected {slide_value}')
