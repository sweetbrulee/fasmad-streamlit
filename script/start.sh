# Make sure that you are at the root of the project

(cd ./.ssl && sudo ../vendor/ssl-proxy/ssl-proxy -from "0.0.0.0:8000" -to "127.0.0.1:8501") &
streamlit run home.py;
(cd ./.ssl && sudo rm -rf *.pem);
sudo pkill -9 ssl-proxy && sudo pkill -9 streamlit;
