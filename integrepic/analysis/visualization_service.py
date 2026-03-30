"""
Visualization Service - Generate interactive Plotly charts for analysis results
"""
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import logging

logger = logging.getLogger(__name__)


class VisualizationService:
    """Service for creating interactive visualizations of analysis data"""

    @staticmethod
    def create_ela_analysis_chart(ela_results):
        """Create ELA analysis visualization"""
        try:
            if not ela_results or isinstance(ela_results, str):
                return None

            # Create ELA metrics chart
            metrics = {
                'Max Difference': ela_results.get('max_difference', 0),
                'Avg Difference': ela_results.get('avg_difference', 0),
            }

            fig = go.Figure(data=[
                go.Bar(
                    x=list(metrics.keys()),
                    y=list(metrics.values()),
                    marker=dict(
                        color=['#ef4444', '#f59e0b'],
                        line=dict(color='rgba(0,0,0,0.1)', width=2)
                    ),
                    text=[f"{v:.2f}" for v in metrics.values()],
                    textposition='outside',
                    hovertemplate='<b>%{x}</b><br>Value: %{y:.2f}<extra></extra>'
                )
            ])

            fig.update_layout(
                title={
                    'text': "Error Level Analysis (ELA) Metrics",
                    'font': {'size': 18, 'color': '#1f2937'}
                },
                xaxis_title="Metric",
                yaxis_title="Value",
                hovermode='x unified',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(255,255,255,0)',
                font=dict(family="Arial, sans-serif", size=12, color="#1f2937"),
                margin=dict(l=50, r=50, t=80, b=50),
                height=400,
            )

            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.2)')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.2)')

            return fig.to_html(include_plotlyjs='cdn', div_id="ela_chart")

        except Exception as e:
            logger.error(f"Error creating ELA chart: {e}")
            return None

    @staticmethod
    def create_image_properties_chart(image_width, image_height, file_size):
        """Create image properties pie chart"""
        try:
            total_pixels = image_width * image_height

            fig = go.Figure(data=[
                go.Pie(
                    labels=['Width', 'Height', 'File Size (KB)'],
                    values=[image_width, image_height, file_size / 1024],
                    marker=dict(
                        colors=['#3b82f6', '#8b5cf6', '#ec4899'],
                        line=dict(color='white', width=2)
                    ),
                    hovertemplate='<b>%{label}</b><br>Value: %{value:.0f}<br>Percentage: %{percent}<extra></extra>'
                )
            ])

            fig.update_layout(
                title={
                    'text': "Image Properties Distribution",
                    'font': {'size': 18, 'color': '#1f2937'}
                },
                paper_bgcolor='rgba(255,255,255,0)',
                font=dict(family="Arial, sans-serif", size=12, color="#1f2937"),
                height=400,
            )

            return fig.to_html(include_plotlyjs=False, div_id="properties_chart")

        except Exception as e:
            logger.error(f"Error creating properties chart: {e}")
            return None

    @staticmethod
    def create_file_info_chart(file_size, image_format, image_width, image_height):
        """Create file information visualization"""
        try:
            fig = go.Figure()

            # Add metrics as text annotations
            metrics_text = f"""
            <b>📋 File Information</b><br><br>
            <b>Format:</b> {image_format.upper()}<br>
            <b>Dimensions:</b> {image_width} × {image_height} px<br>
            <b>File Size:</b> {file_size / (1024*1024):.2f} MB<br>
            <b>Total Pixels:</b> {image_width * image_height:,}
            """

            fig.add_annotation(
                text=metrics_text,
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                bgcolor="rgba(59, 130, 246, 0.1)",
                bordercolor="#3b82f6",
                borderwidth=2,
                borderpad=20,
                font=dict(size=13, color="#1f2937"),
                align="left"
            )

            fig.update_layout(
                title={
                    'text': "File Information Overview",
                    'font': {'size': 18, 'color': '#1f2937'}
                },
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(255,255,255,0)',
                height=300,
                margin=dict(l=20, r=20, t=60, b=20)
            )

            return fig.to_html(include_plotlyjs=False, div_id="fileinfo_chart")

        except Exception as e:
            logger.error(f"Error creating file info chart: {e}")
            return None

    @staticmethod
    def create_steganography_chart(steganography_result):
        """Create steganography detection visualization"""
        try:
            if not steganography_result or isinstance(steganography_result, str):
                return None

            # Check if message was found
            has_message = "detected" in steganography_result.lower() or "found" in steganography_result.lower()

            fig = go.Figure(data=[
                go.Bar(
                    x=['Detection'],
                    y=[1 if has_message else 0],
                    marker=dict(
                        color=['#10b981'] if not has_message else ['#ef4444'],
                        line=dict(color='rgba(0,0,0,0.1)', width=2)
                    ),
                    text=['No Hidden Message Detected' if not has_message else 'Hidden Message Detected'],
                    textposition='outside',
                    hovertemplate='<b>%{text}</b><extra></extra>'
                )
            ])

            fig.update_layout(
                title={
                    'text': "Steganography Detection Status",
                    'font': {'size': 18, 'color': '#1f2937'}
                },
                yaxis_title="Status",
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(255,255,255,0)',
                font=dict(family="Arial, sans-serif", size=12, color="#1f2937"),
                margin=dict(l=50, r=50, t=80, b=50),
                height=300,
            )

            fig.update_yaxes(visible=False)
            fig.update_xaxes(showgrid=False)

            return fig.to_html(include_plotlyjs=False, div_id="stego_chart")

        except Exception as e:
            logger.error(f"Error creating steganography chart: {e}")
            return None

    @staticmethod
    def create_analysis_timeline(created_at, updated_at, analysis_duration):
        """Create analysis timeline visualization"""
        try:
            fig = go.Figure()

            stages = ['Started', 'Processing', 'Completed']
            times = [
                0,
                analysis_duration / 2 if analysis_duration else 0,
                analysis_duration if analysis_duration else 0
            ]

            fig.add_trace(go.Scatter(
                x=stages,
                y=[1, 1, 1],
                mode='lines+markers',
                marker=dict(
                    size=12,
                    color=['#3b82f6', '#f59e0b', '#10b981'],
                    line=dict(color='white', width=2)
                ),
                line=dict(color='#3b82f6', width=3),
                hovertemplate='<b>%{x}</b><br>Time: %{customdata:.2f}s<extra></extra>',
                customdata=times
            ))

            fig.update_layout(
                title={
                    'text': f"Analysis Timeline (Duration: {analysis_duration:.2f}s)" if analysis_duration else "Analysis Timeline",
                    'font': {'size': 18, 'color': '#1f2937'}
                },
                xaxis_title="Stage",
                yaxis_title="Progress",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(255,255,255,0)',
                font=dict(family="Arial, sans-serif", size=12, color="#1f2937"),
                margin=dict(l=50, r=50, t=80, b=50),
                height=300,
                showlegend=False
            )

            fig.update_yaxes(visible=False)

            return fig.to_html(include_plotlyjs=False, div_id="timeline_chart")

        except Exception as e:
            logger.error(f"Error creating timeline chart: {e}")
            return None

    @staticmethod
    def create_metadata_summary(metadata):
        """Create metadata summary visualization"""
        try:
            if not metadata or not isinstance(metadata, dict) or len(metadata) == 0:
                return None

            # Create a summary card with key metadata
            key_fields = ['Make', 'Model', 'DateTime', 'Software', 'Orientation']
            metadata_items = []

            for field in key_fields:
                if field in metadata:
                    metadata_items.append(f"<b>{field}:</b> {str(metadata[field])[:50]}")

            metadata_html = "<br>".join(metadata_items) if metadata_items else "No EXIF metadata found"

            fig = go.Figure()

            fig.add_annotation(
                text=f"<b>📷 EXIF Metadata Summary</b><br><br>{metadata_html}",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                bgcolor="rgba(139, 92, 246, 0.1)",
                bordercolor="#8b5cf6",
                borderwidth=2,
                borderpad=15,
                font=dict(size=11, color="#1f2937"),
                align="left"
            )

            fig.update_layout(
                title={
                    'text': "Metadata Information",
                    'font': {'size': 18, 'color': '#1f2937'}
                },
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(255,255,255,0)',
                height=250,
                margin=dict(l=20, r=20, t=60, b=20)
            )

            return fig.to_html(include_plotlyjs=False, div_id="metadata_chart")

        except Exception as e:
            logger.error(f"Error creating metadata chart: {e}")
            return None

    @staticmethod
    def create_rgb_histogram(image_path):
        """Create RGB color histogram visualization"""
        try:
            from PIL import Image
            import numpy as np

            # Open image and convert to RGB
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # Get pixel data
                pixels = np.array(img)

                # Calculate histograms for each channel
                r_hist = np.histogram(pixels[:, :, 0], bins=256, range=(0, 256))[0]
                g_hist = np.histogram(pixels[:, :, 1], bins=256, range=(0, 256))[0]
                b_hist = np.histogram(pixels[:, :, 2], bins=256, range=(0, 256))[0]

                # Create figure with subplots
                fig = make_subplots(
                    rows=1, cols=3,
                    subplot_titles=('Red Channel', 'Green Channel', 'Blue Channel'),
                    specs=[[{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}]]
                )

                # Red channel
                fig.add_trace(
                    go.Scatter(
                        y=r_hist,
                        fill='tozeroy',
                        line=dict(color='#ef4444', width=2),
                        name='Red',
                        hovertemplate='<b>Intensity: %{x}</b><br>Count: %{y}<extra></extra>'
                    ),
                    row=1, col=1
                )

                # Green channel
                fig.add_trace(
                    go.Scatter(
                        y=g_hist,
                        fill='tozeroy',
                        line=dict(color='#10b981', width=2),
                        name='Green',
                        hovertemplate='<b>Intensity: %{x}</b><br>Count: %{y}<extra></extra>'
                    ),
                    row=1, col=2
                )

                # Blue channel
                fig.add_trace(
                    go.Scatter(
                        y=b_hist,
                        fill='tozeroy',
                        line=dict(color='#3b82f6', width=2),
                        name='Blue',
                        hovertemplate='<b>Intensity: %{x}</b><br>Count: %{y}<extra></extra>'
                    ),
                    row=1, col=3
                )

                fig.update_xaxes(title_text="Pixel Intensity", row=1, col=1)
                fig.update_xaxes(title_text="Pixel Intensity", row=1, col=2)
                fig.update_xaxes(title_text="Pixel Intensity", row=1, col=3)

                fig.update_yaxes(title_text="Frequency", row=1, col=1)
                fig.update_yaxes(title_text="Frequency", row=1, col=2)
                fig.update_yaxes(title_text="Frequency", row=1, col=3)

                fig.update_layout(
                    title={
                        'text': "RGB Color Histogram Analysis",
                        'font': {'size': 18, 'color': '#1f2937'}
                    },
                    height=400,
                    showlegend=True,
                    hovermode='x unified',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(255,255,255,0)',
                    font=dict(family="Arial, sans-serif", size=11, color="#1f2937"),
                    margin=dict(l=50, r=50, t=80, b=50),
                )

                return fig.to_html(include_plotlyjs=False, div_id="rgb_histogram")

        except Exception as e:
            logger.error(f"Error creating RGB histogram: {e}")
            return None

    @staticmethod
    def create_color_distribution_chart(image_path):
        """Create color distribution pie chart showing dominant colors"""
        try:
            from PIL import Image
            import numpy as np
            from collections import Counter

            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # Get pixels and quantize colors
                pixels = np.array(img)
                pixels_reshaped = pixels.reshape(-1, 3)

                # Get unique colors and their counts
                unique_colors, counts = np.unique(
                    pixels_reshaped,
                    axis=0,
                    return_counts=True
                )

                # Get top 10 colors
                top_indices = np.argsort(counts)[-10:][::-1]
                top_colors = unique_colors[top_indices]
                top_counts = counts[top_indices]

                # Create color labels
                color_labels = [f"RGB({r},{g},{b})" for r, g, b in top_colors]
                color_hex = [f"#{int(r):02x}{int(g):02x}{int(b):02x}" for r, g, b in top_colors]

                fig = go.Figure(data=[
                    go.Pie(
                        labels=color_labels,
                        values=top_counts,
                        marker=dict(colors=color_hex),
                        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
                    )
                ])

                fig.update_layout(
                    title={
                        'text': "Top 10 Dominant Colors Distribution",
                        'font': {'size': 18, 'color': '#1f2937'}
                    },
                    height=400,
                    paper_bgcolor='rgba(255,255,255,0)',
                    font=dict(family="Arial, sans-serif", size=11, color="#1f2937"),
                    margin=dict(l=20, r=20, t=80, b=20),
                )

                return fig.to_html(include_plotlyjs=False, div_id="color_distribution")

        except Exception as e:
            logger.error(f"Error creating color distribution chart: {e}")
            return None

    @staticmethod
    def create_color_space_analysis(image_path):
        """Create detailed color space statistics"""
        try:
            from PIL import Image
            import numpy as np

            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                pixels = np.array(img)

                # Calculate statistics for each channel
                r_values = pixels[:, :, 0].flatten()
                g_values = pixels[:, :, 1].flatten()
                b_values = pixels[:, :, 2].flatten()

                stats = {
                    'R': {
                        'mean': np.mean(r_values),
                        'median': np.median(r_values),
                        'std': np.std(r_values),
                        'min': np.min(r_values),
                        'max': np.max(r_values),
                    },
                    'G': {
                        'mean': np.mean(g_values),
                        'median': np.median(g_values),
                        'std': np.std(g_values),
                        'min': np.min(g_values),
                        'max': np.max(g_values),
                    },
                    'B': {
                        'mean': np.mean(b_values),
                        'median': np.median(b_values),
                        'std': np.std(b_values),
                        'min': np.min(b_values),
                        'max': np.max(b_values),
                    }
                }

                # Create visualization with stats
                metrics_text = f"""
                <b>📊 Color Space Statistics</b><br><br>
                <b>Red Channel:</b><br>
                Mean: {stats['R']['mean']:.1f} | Median: {stats['R']['median']:.1f}<br>
                Std Dev: {stats['R']['std']:.1f} | Range: {stats['R']['min']:.0f}-{stats['R']['max']:.0f}<br><br>

                <b>Green Channel:</b><br>
                Mean: {stats['G']['mean']:.1f} | Median: {stats['G']['median']:.1f}<br>
                Std Dev: {stats['G']['std']:.1f} | Range: {stats['G']['min']:.0f}-{stats['G']['max']:.0f}<br><br>

                <b>Blue Channel:</b><br>
                Mean: {stats['B']['mean']:.1f} | Median: {stats['B']['median']:.1f}<br>
                Std Dev: {stats['B']['std']:.1f} | Range: {stats['B']['min']:.0f}-{stats['B']['max']:.0f}
                """

                fig = go.Figure()

                fig.add_annotation(
                    text=metrics_text,
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False,
                    bgcolor="rgba(59, 130, 246, 0.05)",
                    bordercolor="#3b82f6",
                    borderwidth=2,
                    borderpad=15,
                    font=dict(size=10, color="#1f2937", family="monospace"),
                    align="left"
                )

                fig.update_layout(
                    title={
                        'text': "RGB Channel Statistics",
                        'font': {'size': 18, 'color': '#1f2937'}
                    },
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(255,255,255,0)',
                    height=350,
                    margin=dict(l=20, r=20, t=60, b=20)
                )

                return fig.to_html(include_plotlyjs=False, div_id="color_stats")

        except Exception as e:
            logger.error(f"Error creating color space analysis: {e}")
            return None

    @staticmethod
    def create_geolocation_map(metadata, original_filename):
        """Create interactive map from GPS coordinates in EXIF metadata"""
        try:
            import folium
            from folium import plugins

            # Extract GPS coordinates
            gps_info = None
            latitude = None
            longitude = None

            if metadata and 'GPSInfo' in metadata:
                gps_info = metadata['GPSInfo']

            # Try to extract latitude and longitude
            if isinstance(gps_info, dict):
                # Try direct lat/lon keys first
                if 'GPSLatitude' in gps_info and 'GPSLongitude' in gps_info:
                    try:
                        # Parse GPSLatitude (usually a tuple of (degrees, minutes, seconds))
                        lat_data = gps_info['GPSLatitude']
                        lon_data = gps_info['GPSLongitude']

                        # Convert to decimal degrees
                        latitude = float(lat_data[0]) + float(lat_data[1])/60 + float(lat_data[2])/3600
                        longitude = float(lon_data[0]) + float(lon_data[1])/60 + float(lon_data[2])/3600

                        # Apply lat/lon ref (N/S, E/W)
                        if gps_info.get('GPSLatitudeRef') == 'S':
                            latitude = -latitude
                        if gps_info.get('GPSLongitudeRef') == 'W':
                            longitude = -longitude
                    except (TypeError, ValueError, IndexError):
                        pass

            if latitude is None or longitude is None:
                return None

            # Create map centered on the location
            location_map = folium.Map(
                location=[latitude, longitude],
                zoom_start=13,
                tiles='OpenStreetMap'
            )

            # Add marker with popup
            popup_text = f"""
            <b>📸 Image Location</b><br>
            <b>File:</b> {original_filename}<br>
            <b>Latitude:</b> {latitude:.6f}<br>
            <b>Longitude:</b> {longitude:.6f}<br>
            <b>Coordinates:</b> {latitude:.4f}, {longitude:.4f}
            """

            folium.Marker(
                location=[latitude, longitude],
                popup=folium.Popup(popup_text, max_width=300),
                tooltip=f"📸 {original_filename}",
                icon=folium.Icon(color='blue', icon='camera', prefix='fa')
            ).add_to(location_map)

            # Add a circle around the location
            folium.Circle(
                location=[latitude, longitude],
                radius=100,
                popup='~100m radius',
                color='blue',
                fill=False,
                weight=2
            ).add_to(location_map)

            # Convert to HTML string
            map_html = location_map._repr_html_()

            # Wrap in a div with styling
            wrapped_html = f"""
            <div id="gps_map" style="width: 100%; height: 400px; border-radius: 8px; overflow: hidden;">
                {map_html}
            </div>
            <script>
                // Ensure map renders properly
                if (typeof window.mapLoaded === 'undefined') {{
                    window.mapLoaded = true;
                }}
            </script>
            """

            return wrapped_html

        except Exception as e:
            logger.error(f"Error creating geolocation map: {e}")
            return None

    @staticmethod
    def create_location_info_card(metadata):
        """Create location information card from GPS and EXIF data"""
        try:
            if not metadata:
                return None

            gps_info = metadata.get('GPSInfo', {})
            location_data = {}

            # Extract location info
            if isinstance(gps_info, dict):
                location_data = {
                    'latitude': gps_info.get('GPSLatitude', 'N/A'),
                    'longitude': gps_info.get('GPSLongitude', 'N/A'),
                    'altitude': gps_info.get('GPSAltitude', 'N/A'),
                    'timestamp': gps_info.get('GPSTimeStamp', 'N/A'),
                }

            # Extract datetime info
            datetime_str = metadata.get('DateTime', 'Unknown')

            # Create information card
            location_html = f"""
            <div style="padding: 20px; background-color: rgba(59, 130, 246, 0.05); border-left: 4px solid #3b82f6; border-radius: 8px;">
                <h6 style="margin-bottom: 15px; font-weight: bold; color: #1f2937;">
                    <i style="margin-right: 8px;">📍</i>Location Information
                </h6>

                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                    <div>
                        <small style="color: #6b7280; font-weight: bold;">Capture Date/Time</small><br>
                        <span style="color: #1f2937;">{datetime_str}</span>
                    </div>
                    <div>
                        <small style="color: #6b7280; font-weight: bold;">GPS Status</small><br>
                        <span style="color: #10b981;">✓ GPS Data Found</span>
                    </div>
                </div>

                <hr style="border: none; border-top: 1px solid rgba(0,0,0,0.1); margin: 15px 0;">

                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; font-size: 0.9em;">
                    <div>
                        <small style="color: #6b7280; font-weight: bold;">Latitude</small><br>
                        <code style="background-color: #f3f4f6; padding: 5px; border-radius: 4px;">
                            {location_data.get('latitude', 'N/A')}
                        </code>
                    </div>
                    <div>
                        <small style="color: #6b7280; font-weight: bold;">Longitude</small><br>
                        <code style="background-color: #f3f4f6; padding: 5px; border-radius: 4px;">
                            {location_data.get('longitude', 'N/A')}
                        </code>
                    </div>
                    <div>
                        <small style="color: #6b7280; font-weight: bold;">Altitude</small><br>
                        <code style="background-color: #f3f4f6; padding: 5px; border-radius: 4px;">
                            {location_data.get('altitude', 'N/A')}
                        </code>
                    </div>
                    <div>
                        <small style="color: #6b7280; font-weight: bold;">GPS Time</small><br>
                        <code style="background-color: #f3f4f6; padding: 5px; border-radius: 4px;">
                            {location_data.get('timestamp', 'N/A')}
                        </code>
                    </div>
                </div>

                <div style="margin-top: 12px; padding: 10px; background-color: rgba(16, 185, 129, 0.1); border-radius: 4px; border-left: 3px solid #10b981;">
                    <small style="color: #047857;">
                        <i>📌 This image contains embedded GPS coordinates showing the location where the photo was taken.</i>
                    </small>
                </div>
            </div>
            """

            return location_html

        except Exception as e:
            logger.error(f"Error creating location info card: {e}")
            return None
