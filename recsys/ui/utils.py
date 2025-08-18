import re
from io import BytesIO
import os

import requests
import streamlit as st
from PIL import Image, UnidentifiedImageError
import requests
import io
import logging

from recsys import hopsworks_integration
from recsys.config import settings

def print_header(text, font_size=22):
    res = f'<span style="font-size: {font_size}px;">{text}</span>'
    st.markdown(res, unsafe_allow_html=True)


PLACEHOLDER_IMAGE = os.path.join("assets", "placeholder.png")


def fetch_and_process_image(url, size=(300, 200)):
    """
    Fetch an image from a given URL, resize it, and return a PIL Image.
    If fetching fails, return the local placeholder instead.
    """
    try:
        if url and url.startswith("http"):
            resp = requests.get(url, timeout=5)
            resp.raise_for_status()
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            img.thumbnail(size)
            return img
    except Exception as e:
        logging.warning(f"Falling back to placeholder. Failed to fetch {url}: {e}")

    # --- Local fallback ---
    try:
        img = Image.open(PLACEHOLDER_IMAGE).convert("RGB")
        img.thumbnail(size)
        return img
    except Exception as e2:
        logging.error(f"Failed to load local placeholder image: {e2}")
        # Absolute last fallback: generate a blank gray image
        return Image.new("RGB", size, color=(200, 200, 200))

def process_description(description):
    details_match = re.search(r"Details: (.+?)(?:\n|$)", description)
    return details_match.group(1) if details_match else "No details available."


import logging

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "assets")
LOCAL_PLACEHOLDER = os.path.join(ASSETS_DIR, "placeholder.png")

def get_item_image_url(item_id, articles_fv):
    try:
        article_feature_view = articles_fv.get_feature_vector({"article_id": item_id})
        if not article_feature_view:
            raise ValueError("No features found")

        image_url = article_feature_view[-1]
        if not image_url:
            raise ValueError("Empty image URL")

        return image_url
    except Exception as e:
        logging.warning(f"Image not found for article {item_id}: {e}. Using local placeholder.")
        return LOCAL_PLACEHOLDER



@st.cache_resource()
def get_deployments():
    project, fs = hopsworks_integration.get_feature_store()

    ms = project.get_model_serving()

    articles_fv = fs.get_feature_view(
        name="articles",
        version=1,
    )

    query_model_deployment = ms.get_deployment(
        hopsworks_integration.two_tower_serving.HopsworksQueryModel.deployment_name
    )

    ranking_deployment = ms.get_deployment(
        settings.RANKING_MODEL_TYPE
    )

    ranking_deployment.start(await_running=180)
    query_model_deployment.start(await_running=180)

    return articles_fv, ranking_deployment, query_model_deployment
