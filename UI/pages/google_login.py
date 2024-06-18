import streamlit as st
from oauthlib.oauth2 import WebApplicationClient
import os
import requests

# Google OAuth 2.0 Configuration
GOOGLE_CLIENT_ID = "539506164308-5du69hk8kn8p0q3uodi5fte4vn2pppjj.apps.googleusercontent.com"
GOOGLE_CLIENT_SECRET = "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"
GOOGLE_DISCOVERY_URL = "https://accounts.google.com/.well-known/openid-configuration"

# Allow insecure transport for local development
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

# Get Google's authorization endpoint
def get_google_provider_cfg():
    return requests.get(GOOGLE_DISCOVERY_URL).json()

# Set up the OAuth client
client = WebApplicationClient(GOOGLE_CLIENT_ID)

# Streamlit App
st.title("Streamlit Google Login")

# Check if the user is logged in
if "email" not in st.session_state:
    google_provider_cfg = get_google_provider_cfg()
    authorization_endpoint = google_provider_cfg["authorization_endpoint"]

    redirect_uri = "http://localhost:8501/chatbot"
    request_uri = client.prepare_request_uri(
        authorization_endpoint,
        redirect_uri=redirect_uri,
        scope=["openid", "email", "profile"],
    )

    st.write("## Login with Google")
    st.markdown(f"[Login with Google]({request_uri})")

    # After the user logs in, Google will redirect them back to the redirect_uri with a code.
    query_params = st.experimental_get_query_params()
    if "code" in query_params:
        code = query_params["code"][0]
        try:
            token_url = google_provider_cfg["token_endpoint"]
            token_params = {
                "code": code,
                "redirect_uri": redirect_uri,
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
            }
            body = client.prepare_request_body(**token_params)
            headers = {'Content-Type': 'application/x-www-form-urlencoded'}
            token_response = requests.post(token_url, headers=headers, data=body)

            token_response.raise_for_status()
            token_data = token_response.json()

            # Store token_data in session state
            st.session_state["token_data"] = token_data

            # Verify token response and fetch user info if successful
            if 'access_token' in token_data:
                userinfo_endpoint = google_provider_cfg["userinfo_endpoint"]
                headers = {'Authorization': f'Bearer {token_data["access_token"]}'}
                userinfo_response = requests.get(userinfo_endpoint, headers=headers)

                userinfo_response.raise_for_status()
                userinfo_data = userinfo_response.json()

                if userinfo_data.get("email_verified"):
                    st.session_state["email"] = userinfo_data["email"]
                    st.session_state["name"] = userinfo_data["name"]
                    st.session_state["picture"] = userinfo_data["picture"]
                    st.experimental_rerun()  # Rerun to update the state and refresh the UI
                else:
                    st.error("User email not available or not verified by Google.")
            else:
                st.error("Token request did not succeed.")

        except requests.exceptions.HTTPError as e:
            st.error(f"HTTP error occurred: {e.response.text}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
else:
    st.write(f"Logged in as {st.session_state['name']}")
    st.image(st.session_state["picture"])
    st.write(f"Email: {st.session_state['email']}")

    if st.button("Logout"):
        try:
            # Check if token_data exists in session state
            if "token_data" in st.session_state:
                token_data = st.session_state["token_data"]
                # Revoke token to invalidate it
                if 'access_token' in token_data:
                    revoke_url = 'https://oauth2.googleapis.com/revoke'
                    revoke_params = {'token': token_data['access_token']}
                    revoke_response = requests.post(revoke_url, params=revoke_params)

                    if revoke_response.status_code == 200:
                        st.success("Successfully logged out.")
                    else:
                        st.error(f"Failed to revoke token: {revoke_response.text}")
                else:
                    st.error("Access token not found in token_data.")
            else:
                st.error("Token data not found in session state.")
        
            # Clear session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            
            st.experimental_rerun()  # Rerun to update the state and refresh the UI
        
        except requests.exceptions.HTTPError as e:
            st.error(f"HTTP error occurred during logout: {e.response.text}")
        except Exception as e:
            st.error(f"An error occurred during logout: {e}")
