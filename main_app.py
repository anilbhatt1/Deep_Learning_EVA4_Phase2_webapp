import streamlit as st

def main():
    """ EVA4-Phase 2 Deep Learning Models """
    local_css("style.css")
    st.markdown("<h1 style='text-align: center; color: black;font-size: 40px;'>Neural Eyes</h1>", unsafe_allow_html=True)
    st.text('')
    st.text('')

    #Mobilenet
    if st.checkbox("Predict flying objects"):
        st.subheader("Predicting flying objects using mobilenet")


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


if __name__ == '__main__':
    main()
