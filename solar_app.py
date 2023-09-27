import streamlit as st
import geocoder
import requests
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

BING_KEY = st.secrets.BING_KEY
GOOGLE_SOLAR_KEY = st.secrets.GOOGLE_SOLAR_KEY
SOLAR_INSIGHTS_ENDPOINT = 'https://solar.googleapis.com/v1/buildingInsights:findClosest?location.latitude={}&location.longitude={}&requiredQuality=LOW&key={}'

# streamlit_app.py

import streamlit as st

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True

@st.cache_data
def get_lat_lng(address):
    g = geocoder.bing(address, key=BING_KEY)
    return g.latlng

@st.cache_data
def get_solar_insights(lat, lng):
    response = requests.get(SOLAR_INSIGHTS_ENDPOINT.format(lat, lng, GOOGLE_SOLAR_KEY))
    return response.json()

def get_google_maps_image(lat, lon, zoom=20, size="600x600", maptype="satellite", api_key=GOOGLE_SOLAR_KEY):
    base_url = "https://maps.googleapis.com/maps/api/staticmap?"
    params = {
        "center": f"{lat},{lon}",
        "zoom": zoom,
        "size": size,
        "maptype": maptype,
        "key": api_key
    }
    response = requests.get(base_url, params=params)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content))
    return image

# calculator

def get_yearly_energy(data, panels_count):
    for config in data['solarPotential']['solarPanelConfigs']:
        if config['panelsCount'] == panels_count:
            return config['yearlyEnergyDcKwh']
    return None

def solar_calculator(data):
    st.subheader('Solar Savings Calculator')

    # Extract min and max panels count from the API results
    min_panels = data['solarPotential']['solarPanelConfigs'][0]['panelsCount']
    max_panels = data['solarPotential']['maxArrayPanelsCount']

    # Input widgets with default values
    #if 'num_panels' not in st.session_state:
     #   st.session_state.num_panels = max_panels

    if 'kw_per_panel' not in st.session_state:
        st.session_state.kw_per_panel = 0.3

    if 'electricity_price' not in st.session_state:
        st.session_state.electricity_price = 0.12

    if 'panel_area_m2' not in st.session_state:
        st.session_state.panel_area_m2 = 1.5

    #st.session_state.num_panels = st.number_input('Number of Solar Panels', value=st.session_state.num_panels)
    #st.session_state.kw_per_panel = st.number_input('kW per Solar Panel', value=st.session_state.kw_per_panel)
    #st.session_state.electricity_price = st.number_input('Price of Electricity (per kWh)', value=st.session_state.electricity_price)
    #st.session_state.panel_area_m2 = st.number_input('Area per Solar Panel (in m^2)', value=st.session_state.panel_area_m2)

    # Calculations
    # Inputs
    panels = st.slider('Number of Solar Panels', min_value=min_panels, max_value=max_panels, value=max_panels)
    user_wattage = st.number_input('Wattage of Solar Panel', value=430.0, step=10.0)  # User can adjust wattage
    electricity_price = st.number_input('Price of Electricity (€/kWh)', value=0.40, step=0.01)

    # Calculate
    api_energy = get_yearly_energy(data, panels)
    adjusted_energy = api_energy * (user_wattage / 250.0)  # Adjust based on user-specified wattage
    yearly_savings = adjusted_energy * electricity_price

    # Display
    st.write(f"Estimated Yearly Energy Generation: {adjusted_energy:.2f} kWh")
    st.write(f"Estimated Yearly Savings: €{yearly_savings:.2f}")

def main():
    st.title('Solar Panel Insights')

    address = st.text_input("Enter your address:")

    # Check if the address is in session_state
    if 'address' not in st.session_state:
        st.session_state.address = ''

    # Check if data is already in session_state and if the address has changed
    if 'data' not in st.session_state or st.session_state.address != address:
        if st.button('Get Insights'):
            lat, lng = get_lat_lng(address)
            data = get_solar_insights(lat, lng)
            st.session_state.data = data
            st.session_state.address = address

    # If data is in session_state, display it
    if 'data' in st.session_state:
        data = st.session_state.data
        lat, lng = get_lat_lng(st.session_state.address)

        # Display the image of the house
        image = get_google_maps_image(lat, lng)
        st.image(image, caption=f"House Image from {data['imageryDate']['year']}-{data['imageryDate']['month']}-{data['imageryDate']['day']}", use_column_width=True)

        # Display solar data
        st.subheader('Solar Potential')
        # Add a note using st.markdown
        st.markdown("_Disclaimer: Based on Google Solar API data (panel height - 1.65m, panel width - 0.99m, 250 Watts)._")
        st.markdown(f"_Coordinates of the address: {lat} {lng}_")
        st.write(f"Max Array Panels Count: {data['solarPotential']['maxArrayPanelsCount']}")
        st.write(f"Max Solar Panel Area (m^2): {data['solarPotential']['maxArrayAreaMeters2']}")
        st.write(f"Total Roof Area (m^2): {data['solarPotential']['wholeRoofStats']['areaMeters2']}")
        st.write(f"Roof Segments: {len(data['solarPotential']['roofSegmentStats'])}")
        st.write(f"Max Sunshine Hours Per Year: {data['solarPotential']['maxSunshineHoursPerYear']}")

        # Calculator
        solar_calculator(data)

if __name__ == '__main__':
    if check_password():
        main()
