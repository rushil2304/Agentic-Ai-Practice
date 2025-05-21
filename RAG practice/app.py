import gradio as gr
import random
from smolagents import GradioUI, CodeAgent, HfApiModel
from tools import DuckDuckGoSearchTool, WeatherInfoTool, HubStatsTool
from retriever import load_guest_dataset


model = HfApiModel()


search_tool = DuckDuckGoSearchTool()


weather_info_tool = WeatherInfoTool()


hub_stats_tool = HubStatsTool()


guest_info_tool = load_guest_dataset()


alfred = CodeAgent(
    tools=[guest_info_tool, weather_info_tool, hub_stats_tool, search_tool], 
    model=model,
    add_base_tools=True,  # Add any additional base tools
    planning_interval=3   # Enable planning every 3 steps
)

if __name__ == "__main__":
    GradioUI(alfred).launch()