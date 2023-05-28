import streamlit as st


def register_experimental_rerun_button():
    """Register the use of the experimental rerun button.

    Example
    -------
    ```python
    (In some page...)
    >>> register_experimental_rerun_button()

    (NOT)
    >>> if register_experimental_rerun_button():
    ...     # do something...
    ```
    """

    if st.button("重新运行 (实验性)"):
        st.experimental_rerun()


def register_clear_all_cache_button():
    """Register the use of the clear all cache button.

    Example
    -------
    ```python
    (In some page...)
    >>> register_clear_all_cache_button()

    (NOT)
    >>> if register_clear_all_cache_button():
    ...     # do something...
    ```
    """

    if st.button("清除所有缓存 (不建议)"):
        st.cache_data.clear()
        st.cache_resource.clear()
