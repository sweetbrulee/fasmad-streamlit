import streamlit as st


def mount_experimental_rerun_button():
    """Mount the experimental rerun button on the page.

    Example
    -------
    ```python
    (In some page...)
    >>> mount_experimental_rerun_button()

    (NOT)
    >>> if mount_experimental_rerun_button():
    ...     # do something...
    ```
    """

    if st.button("重新运行 (实验性)"):
        st.experimental_rerun()


def mount_clear_all_cache_button():
    """Mount the clear all cache button on the page.

    Example
    -------
    ```python
    (In some page...)
    >>> mount_clear_all_cache_button()

    (NOT)
    >>> if mount_clear_all_cache_button():
    ...     # do something...
    ```
    """

    if st.button("清除所有缓存 (不建议)"):
        st.cache_data.clear()
        st.cache_resource.clear()
